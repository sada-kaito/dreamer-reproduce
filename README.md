# 説明
**Dream to Control: Learning Behaviors by Latent Imagination (Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi, 2020)**  のDreamerを再実装しました．  
Atari環境は用いず，dm_control suite環境のみで動かせます．

# 実行環境
windows 11  
rtx a4000　または　Geforce rtx 3060Ti

# 環境とライブラリ
**anacondaで仮想環境作成**
anaconda prompt上でdreamer-gpu.ymlを保存したディレクトリに移動して以下を入力．
```
conda env create -n 環境名 -f dreamer-gpu.yml
```
**もしくは以下のライブラリをインストール**
- python                   3.9.18
- pandas                   2.1.3
- matplotlib               3.8.2
- numpy                    1.26.2
- tensorflow-gpu           2.10.0
- tensorflow_probability   0.18.0
- dm-control               1.0.15
- gym                      0.26.2
- cuda-toolkit             11.2
- cudnn                    8.1

# 既知のエラー(修正方法が分からなかった)
import egl errorみたいなエラーがたまに出ます．  
**このエラーが出た時の対処法**
僕はspyderで動かしていたので，その時の対処法を記しておきます．  
spyderを一度閉じるもしくはPCの再起動を行う．  
spyderを開き，test/t_dm_control.pyを開いて，一度実行した後に，dreamer_repro.pyを実行する．  
