# P.137 リスト4.2: 4,5,6,7/smiles_vocab.py

import torch
from torch import nn


class SmilesVocabulary(object):
    """SMILES 系列を整数列に変換するためのクラス

    Attributes:
        pad (str): 系列の長さを揃えるために終了記号の後に追加する空白文字
        sos (str): 開始記号〈sos〉
        eos (str): 終了記号〈eos〉
        pad_idx (int): padding のインデックス
        sos_idx (int): sos のインデックス
        eos_idx (int): eos のインデックス
        char_list (list[str]): SMILES に含まれる文字と padding, sos, eos を含むリスト
    """

    pad: str = " "
    sos: str = "!"
    eos: str = "?"
    pad_idx: int = 0
    sos_idx: int = 1
    eos_idx: int = 2

    def __init__(self):
        # SMILES に含まれる文字と padding, sos, eos を含むリスト
        self.char_list: list[str] = [self.pad, self.sos, self.eos]

    def update(self, smiles: str) -> torch.Tensor:
        """SMILES 系列を受け取り、char_list を更新した上で、smiles2seq() によって対応する整数列を返す

        Args:
            smiles (str): SMILES 系列

        Returns:
            torch.Tensor: SMILES に対応する整数列
        """
        char_set = set(smiles)
        char_set = char_set - set(self.char_list)
        self.char_list.extend(sorted(list(char_set)))
        return self.smiles2seq(smiles)

    def smiles2seq(self, smiles: str) -> torch.Tensor:
        """SMILES 系列を受け取り、対応する整数列を返す

        Args:
            smiles (str): SMILES 系列

        Returns:
            torch.Tensor: SMILES に対応する整数列"""
        return torch.tensor(
            [self.sos_idx]
            + [self.char_list.index(each_char) for each_char in smiles]
            + [self.eos_idx]
        )

    def seq2smiles(self, seq: torch.Tensor, wo_special_char: bool = True) -> str:
        """整数系列 seq を受け取り、対応する SMILES 系列を返す

        Args:
            seq (torch.Tensor): SMILES に変換したい整数列
            wo_special_char (bool, optional): True のとき特殊文字 (padding, sos, eos) を除く.
                Defaults to True.

        Returns:
            str: SMILES 系列
        """
        if wo_special_char:
            return self.seq2smiles(
                # padding, sos, eos を除いた系列を取得
                seq[
                    torch.where(
                        (seq != self.pad_idx)
                        * (seq != self.sos_idx)
                        * (seq != self.eos_idx)
                    )
                ],
                wo_special_char=False,
            )
        return "".join([self.char_list[each_idx] for each_idx in seq])

    def batch_update(self, smiles_list: list[str]) -> tuple[torch.Tensor, list[str]]:
        """SMILES 系列のリストを受け取り、それぞれに update() を適用した上で、空文字を加えて長さをそろえた整数系列を返す

        Args:
            smiles_list (list[str]): SMILES 系列のリスト

        Returns:
            tuple[torch.Tensor, list[str]]: SMILES に対応する整数系列のテンソルと、元の SMILES 系列のリスト
        """
        seq_list: list[torch.Tensor] = []  # 返す整数系列のリスト
        out_smiles_list = []  # 元の SMILES 系列（改行文字削除）のリスト
        for each_smiles in smiles_list:
            if each_smiles.endswith("\n"):
                each_smiles = each_smiles.strip()
            # 使用文字を更新して、SMILES を格納
            seq_list.append(self.update(each_smiles))
            # SMILES の整数系列を格納
            out_smiles_list.append(each_smiles)
        # バッチサイズ * 最大系列長 のテンソルになるように padding
        right_padded_batch_seq = nn.utils.rnn.pad_sequence(
            seq_list, batch_first=True, padding_value=self.pad_idx
        )
        return right_padded_batch_seq, out_smiles_list

    def batch_update_from_file(
        self, file_path: str, with_smiles: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[str]]:
        """SMILES 系列のデータセットのファイルを受け取り、batch_update() を適用した結果を返す

        Args:
            file_path (str): SMILES 系列のデータセットのファイルパス
            with_smiles (bool, optional): True のとき、SMILES 系列のリストも返す. Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list[str]]: SMILES に対応する整数系列のテンソル、
                または、SMILES に対応する整数系列のテンソルと、元の SMILES 系列のリスト
        """
        seq_tensor, smiles_list = self.batch_update(open(file_path).readlines())
        if with_smiles:
            return seq_tensor, smiles_list
        return seq_tensor
