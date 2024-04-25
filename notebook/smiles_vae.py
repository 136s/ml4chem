# P.172 リスト5.1: 4,5,6,7/smiles_vae.py

from typing import Any
import torch
from torch import nn
from torch.distributions import Categorical
from smiles_vocab import SmilesVocabulary


class SmilesVAE(nn.Module):

    def __init__(
        self,
        vocab: SmilesVocabulary,
        latent_dim: int,
        emb_dim: int = 128,
        max_len: int = 100,
        encoder_params: dict[str, Any] = {
            "hidden_size": 128,
            "num_layers": 1,
            "dropout": 0.0,
        },
        decoder_params: dict[str, Any] = {
            "hidden_size": 128,
            "num_layers": 1,
            "dropout": 0.0,
        },
        encoder2out_params: dict[str, Any] = {"out_dim_list": [128, 128]},
    ):
        """SMILES の文字列を生成する VAE モデル

        Args:
            vocab (SmilesVocabulary): SMILES の文字列を生成するためのアルファベット集合
            latent_dim (int): 潜在空間の次元数
            emb_dim (int, optional): 埋め込みベクトルの次元数. Defaults to 128.
            max_len (int, optional): SMILES 系列の生成時に許す最大の系列長. Defaults to 100.
            encoder_params (dict[str, Any], optional): エンコーダの LSTM のパラメータ.
                Defaults to {"hidden_size": 128, "num_layers": 1, "dropout": 0.0}.
            decoder_params (dict[str, Any], optional): デコーダの LSTM のパラメータ.
                Defaults to {"hidden_size": 128, "num_layers": 1, "dropout": 0.0}.
            encoder2out_params (dict[str, Any], optional): エンコーダの LSTM の出力を正規分布のパラメタに変換する多層ニューラルネットワークのハイパーパラメータ.
                Defaults to {"out_dim_list": [128, 128]}.
        """
        super().__init__()
        self.vocab: SmilesVocabulary = vocab
        vocab_size = len(self.vocab.char_list)
        self.max_len: int = max_len
        self.latent_dim: int = latent_dim
        self.beta: float = 1.0
        # 埋め込みベクトル
        self.embedding: nn.Embedding = nn.Embedding(
            vocab_size, emb_dim, padding_idx=vocab.pad_idx
        )
        # エンコーダ
        self.encoder: nn.LSTM = nn.LSTM(emb_dim, batch_first=True, **encoder_params)
        # エンコーダの LSTM の出力を変換する多層ニューラルネットワーク
        self.encoder2out: nn.Sequential = nn.Sequential()
        in_dim = (
            encoder_params["hidden_size"] * 2
            if encoder_params.get("bidirectional", False)
            else encoder_params["hidden_size"]
        )
        for each_out_dim in encoder2out_params["out_dim_list"]:
            self.encoder2out.append(nn.Linear(in_dim, each_out_dim))
            self.encoder2out.append(nn.Sigmoid())
            in_dim = each_out_dim
        # self.encoder2out の出力を潜在空間上の正規分布の平均に変換する線形モデル
        self.encoder_out2mu: nn.Linear = nn.Linear(in_dim, latent_dim)
        # self.encoder2out の出力を潜在空間上の正規分布の分散共分散行列の対角成分に変換する線形モデル
        self.encoder_out2logvar: nn.Linear = nn.Linear(in_dim, latent_dim)
        # 潜在ベクトルを、デコーダの LSTM の隠れ状態に変換するモデル
        self.latent2dech: nn.Linear = nn.Linear(
            latent_dim, decoder_params["hidden_size"] * decoder_params["num_layers"]
        )
        # 潜在ベクトルを、デコーダの LSTM の細胞状態に変換するモデル
        self.latent2decc: nn.Linear = nn.Linear(
            latent_dim, decoder_params["hidden_size"] * decoder_params["num_layers"]
        )
        self.latent2emb: nn.Linear = nn.Linear(latent_dim, emb_dim)
        # デコーダ
        self.decoder: nn.LSTM = nn.LSTM(
            emb_dim, batch_first=True, bidirectional=False, **decoder_params
        )
        # デコーダの出力を、アルファベット空間上のロジットベクトルに変換するモデル
        self.decoder2vocab: nn.Linear = nn.Linear(
            decoder_params["hidden_size"], vocab_size
        )
        # デコーダの出力の確率分布
        self.out_dist_cls: type[torch.distributions.distribution.Distribution] = (
            Categorical
        )
        # 損失関数
        self.loss_func: nn.CrossEntropyLoss = nn.CrossEntropyLoss(reduction="none")

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, in_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """SMILES 系列を潜在空間上の正規分布の平均と分散共分散行列の対角成分の対数にエンコード

        Args:
            in_seq (torch.Tensor): SMILES 系列の整数値テンソル

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 潜在ベクトルの平均と分散共分散行列の対角成分
        """
        # in_seq (batch_size, seq_len) を埋め込みベクトルの行列に変換
        in_seq_emb: torch.Tensor = self.embedding(in_seq)
        # 埋め込みベクトルの系列をエンコーダに入力
        # 隠れ状態の系列 out_seq: サンプルサイズ * 系列長 * 隠れ状態の次元
        # 最終隠れ状態 (h, c)
        out_seq, (h, c) = self.encoder(in_seq_emb)
        # 末尾の隠れ状態は、入力系列すべてを反映した隠れ状態であり、これを使ってエンコーダの出力を作る
        last_out: torch.Tensor = out_seq[:, -1, :]
        out: torch.Tensor = self.encoder2out(last_out)
        # 潜在空間上の正規分布の平均と分散共分散行列をつくり、エンコーダの出力とする
        return (self.encoder_out2mu(out), self.encoder_out2logvar(out))

    def reparam(
        self, mu: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """再パラメタ化法の適用

        Args:
            mu (torch.Tensor): 潜在空間上の正規分布の平均
            logvar (torch.Tensor): 潜在空間上の正規分布の分散共分散行列の対角成分の対数
            deterministic (bool, optional): False のとき確率的なサンプリングを行う. Defaults to False.

        Returns:
            torch.Tensor: 潜在ベクトル（バッチサイズ * 潜在空間の次元数）
        """
        # 潜在空間上の正規分布の分散共分散行列の対角成分の対数から標準偏差を計算
        std = torch.exp(0.5 * logvar)
        # 標準偏差を何倍にするかを正規分布からランダムサンプリング
        eps = torch.randn_like(std)
        if deterministic:
            return mu
        else:
            return mu + std * eps

    def decode(
        self,
        z: torch.Tensor,
        out_seq: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """潜在ベクトルを SMILES 系列にデコード

        Args:
            z (torch.Tensor): 潜在ベクトル（バッチサイズ * 潜在空間の次元数）
            out_seq (torch.Tensor, optional): デコードの正解系列. 正解の SMILES 系列がない場合のデコードは None. Defaults to None.
            deterministic (bool, optional): False のとき確率的なサンプリングを行う. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: デコードの出力のロジットベクトルとデコードの正解系列
        """
        batch_size: int = z.shape[0]
        # デコードに用いる LSTM の隠れ状態 h と細胞状態 c を潜在ベクトルから作成
        h_unstructured: torch.Tensor = self.latent2dech(z)
        c_unstructured: torch.Tensor = self.latent2decc(z)
        h: torch.Tensor = torch.stack(
            [
                h_unstructured[:, each_idx : each_idx + self.decoder.hidden_size]
                for each_idx in range(
                    0, h_unstructured.shape[1], self.decoder.hidden_size
                )
            ]
        )
        c: torch.Tensor = torch.stack(
            [
                c_unstructured[:, each_idx : each_idx + self.decoder.hidden_size]
                for each_idx in range(
                    0, c_unstructured.shape[1], self.decoder.hidden_size
                )
            ]
        )
        if out_seq is None:
            # 正解の SMILES 系列が無い場合は、LSTM から生成されたアルファベットを再び LSTM に入力して
            # 繰り返しアルファベットを生成することで SMILES 系列を生成し、SMILES 系列とその対数尤度を返す
            with torch.no_grad():
                # バッチごとに検証データの損失関数の値を計算（ここでは勾配の計算をしない）
                in_seq = torch.tensor(
                    [[self.vocab.sos_idx]] * batch_size, device=self.device
                )
                out_logit_list = []
                for each_idx in range(self.max_len):
                    # 入力系列を埋め込みベクトルの系列に変換
                    in_seq_emb = self.embedding(in_seq)
                    # デコーダに入力して出力系列を得る
                    out_seq, (h, c) = self.decoder(in_seq_emb[:, -1:, :], (h, c))
                    # アルファベット集合上の確率ベクトルのロジット値に変換
                    out_logit = self.decoder2vocab(out_seq)
                    out_logit_list.append(out_logit)
                    if deterministic:
                        # 確率的なサンプリングを行わない場合は、最大値を取る
                        out_idx = torch.argmax(out_logit, dim=2)
                    else:
                        # 確率的なサンプリングを行う場合は、カテゴリカル分布からサンプリング
                        out_prob = nn.functional.softmax(out_logit, dim=2)
                        out_idx = self.out_dist_cls(probs=out_prob).sample()
                    # 系列の末尾にアルファベットを追加して、最大系列長になるまで繰り返す
                    in_seq = torch.cat((in_seq, out_idx), dim=1)
                # SMILES 系列やそのロジット値の配列を返す
                return torch.cat(out_logit_list, dim=1), in_seq
        else:
            # 正解の SMILES 系列がある場合は、正解の SMILES 系列をデコードして対数尤度を返す
            # 埋め込みベクトルの系列に変換
            out_seq_emb: torch.Tensor = self.embedding(out_seq)
            out_seq_emb_out, _ = self.decoder(out_seq_emb, (h, c))
            # 対数尤度（バッチサイズ * 系列長 * アルファベット数）を計算
            out_seq_vocab_logit: torch.Tensor = self.decoder2vocab(out_seq_emb_out)
            # 損失関数の計算に使われるため、系列長を 1 短くしている
            return out_seq_vocab_logit[:, :-1], out_seq[:-1]

    def forward(
        self,
        in_seq: torch.Tensor,
        out_seq: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """エンコードとデコードを行い、損失関数の計算に必要な値を算出

        Args:
            in_seq (torch.Tensor): 入力系列
            out_seq (torch.Tensor, optional): 出力系列. Defaults to None.
            deterministic (bool, optional): False のとき確率的なサンプリングを行う. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: デコードされた SMILES 系列のロジット値、
                潜在ベクトルの平均、潜在ベクトルの分散共分散行列の対角成分の対数
        """
        mu, logvar = self.encode(in_seq)
        # 正規分布に対して再パラメタ化
        z = self.reparam(mu, logvar, deterministic=deterministic)
        out_seq_logit, _ = self.decode(z, out_seq, deterministic=deterministic)
        return out_seq_logit, mu, logvar

    def loss(self, in_seq: torch.Tensor, out_seq: torch.Tensor) -> torch.Tensor:
        """損失関数の計算

        Args:
            in_seq (torch.Tensor): 入力系列（バッチサイズ * 系列長）
            out_seq (torch.Tensor): 出力系列（バッチサイズ * 系列長）

        Returns:
            torch.Tensor: 損失関数の値
        """
        # モデルの出力を計算
        out_seq_logit, mu, logvar = self.forward(in_seq, out_seq)
        # 交差エントロピー損失を計算
        neg_likelihood: torch.Tensor = self.loss_func(
            out_seq_logit.transpose(1, 2), out_seq[:, 1:]
        )
        neg_likelihood: torch.Tensor = neg_likelihood.sum(axis=1).mean()
        # KL 情報量を計算
        kl_div: torch.Tensor = (
            -0.5 * (1.0 + logvar - mu**2 - torch.exp(logvar)).sum(axis=1).mean()
        )
        # β-VAE のため、KL 情報量に β を乗じている
        return neg_likelihood + self.beta * kl_div

    def generate(
        self,
        z: torch.Tensor | None = None,
        sample_size: int | None = None,
        deterministic: bool = False,
    ) -> list[str]:
        """デコーダを用いて SMILES 系列を生成

        Args:
            z (torch.Tensor, optional): 潜在ベクトル. Defaults to None.
            sample_size (int, optional): 生成する SMILES 系列の数. Defaults to None.
            deterministic (bool, optional): False のとき確率的なサンプリングを行う. Defaults to False.

        Returns:
            list[str]: 生成された SMILES 系列のリスト
        """
        device = next(self.parameters()).device
        if z is None:
            # 潜在ベクトルが与えられない場合は、正規分布からサンプリング
            z = torch.randn(sample_size, self.latent_dim).to(device)
        else:
            z = z.to(device)
        with torch.no_grad():
            # ネットワークを推論モードにする
            self.eval()
            # デコードを行い、SMILES 系列を生成
            _, out_seq = self.decode(z, deterministic=deterministic)
            out = [self.vocab.seq2smiles(each_seq) for each_seq in out_seq]
            # ネットワークを訓練モードにする
            self.train()
            return out

    def reconstruct(
        self,
        in_seq: torch.Tensor,
        deterministic: bool = True,
        max_reconstruct: int | None = None,
        verbose: bool = True,
    ) -> list[bool]:
        """再構成成功率の計算

        Args:
            in_seq (torch.Tensor): SMILES 系列の集合
            deterministic (bool, optional): False のとき確率的なサンプリングを行う. Defaults to True.
            max_reconstruct (int, optional): 再構成する系列の最大数. Defaults to None.
            verbose (bool, optional): True のとき、入力した SMILES 系列と再構成した SMILES 系列を表示する. Defaults to True.

        Returns:
            list[bool]: 再構成成功の真偽値のリスト
        """
        # ネットワークを推論モードにする
        self.eval()
        if max_reconstruct is not None:
            in_seq = in_seq[:max_reconstruct]
        mu, logvar = self.encode(in_seq)
        z = self.reparam(mu, logvar, deterministic=deterministic)
        _, out_seq = self.decode(z, deterministic=deterministic)

        success_list: list[bool] = []
        for each_idx, each_seq in enumerate(in_seq):
            truth = self.vocab.seq2smiles(each_seq)[::-1]
            pred = self.vocab.seq2smiles(out_seq[each_idx])
            success_list.append(truth == pred)
            if verbose:
                print("{}\t{} -> {}".format(truth == pred, truth, pred))
        # ネットワークを訓練モードにする
        self.train()
        return success_list
