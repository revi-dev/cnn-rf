import pathlib


class CNNReceptiveField:
    def __init__(self, layers: dict[str, tuple] | tuple[tuple]):
        """ コンストラクタ
        
        Parameters
        ----------
        layers : dict[str, tuple] | tuple[tuple]
            CNNレイヤーの定義
            
            key  : レイヤー名
            value: (カーネルサイズ, ストライド, パディング)
        """
        if isinstance(layers, dict):
            self.layers = layers
        else:
            self.layers = {f'layer_{i}':layer for i, layer in enumerate(layers)}
            
        self._headers = ('Layer', 'Kernel', 'Stride', 'Padding', 
                         'InputSize', 'OutputSize', 'ReceptiveField')
        self._layers = []

    def calculate(self, input_size: int=224) -> None:
        """ CNNの受容野を計算する

        Parameters
        ----------
        input_size : int, optional
            入力画像サイズ, by default 224
        """
        self._layers = []
        self._header_width = [len(header)+2 for header in self._headers]
        rf = 1
        s_prod = 1
        
        for layer_name, (kernel, stride, padding) in self.layers.items():
            self._header_width[0] = max(self._header_width[0], len(layer_name)+2)
            _layer = (layer_name, kernel, stride, padding, input_size)
            
            input_size = (input_size + 2 * padding - kernel) // stride + 1
            rf += (kernel - 1) * s_prod
            s_prod *= stride
            
            _layer += (input_size, rf)
            self._layers.append(_layer)

    def show_layers(self) -> None:
        """ CNNレイヤーの情報を表示する
        
        このメソッドを呼び出す前にcalculateメソッドを実行する必要がある.
        """
        print(self._to_markdown())
        print()
        
    def to_markdown(self, filepath: str | pathlib.Path) -> None:
        """ CNNレイヤーの情報をmarkdownファイルに保存する.

        Parameters
        ----------
        filepath : str | pathlib.Path
            _description_
        """
        markdown = self._to_markdown()
        
        p = pathlib.Path(filepath)
        p.parent.mkdir(exist_ok=True, parents=True)
        
        with open(p, 'w') as f:
            f.write(markdown)

    def _to_markdown(self) -> str:
        """ CNNレイヤーの情報をmarkdown形式のテキストに変換する.

        Returns
        -------
        str
            markdownテキスト
        """
        header_row = [f'{name:^{width}}' for name, width in zip(self._headers, self._header_width)]
        header_row = '|' + '|'.join(header_row) + '|\n'
        
        align_row = ['-'*(width-1) + ':' for width in self._header_width]
        align_row = '|' + '|'.join(align_row) + '|\n'
        
        layer_rows = [[f'{value:>{width-1}} ' for value, width in zip(columns, self._header_width)]
                      for columns in self._layers]
        layer_rows = '\n'.join(['|' + '|'.join(layer_row) + '|' for layer_row in layer_rows])
        
        return header_row + align_row + layer_rows
  