# 説明
**Dream to Control: Learning Behaviors by Latent Imagination (Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi, 2020)**  のDreamerを再実装しました．  

  
![walk_animation](https://github.com/sada-kaito/dreamer-reproduce/assets/143638502/ea0ff36c-e5a2-4ccb-9f9a-e27883a3a8bd)
![run_animation](https://github.com/sada-kaito/dreamer-reproduce/assets/143638502/bcb1c6b8-36b4-41f7-b8ef-9e29e7d7cf4a)  

Atari環境は用いず，dm_control suite環境のみで動かせます．

# 実行環境
windows 11  
rtx a4000　または　Geforce rtx 3060Ti  
anaconda  
spyder  

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

# ~~既知のエラー(修正方法が分からなかった)~~　修正済み
dreamer_repro.pyであらかじめdm_control suiteをインポートしておくとエラーが起きなくなりました。
```
from dm_control import suite
```

![スクリーンショット 2024-01-11 102131](https://github.com/sada-kaito/dreamer-reproduce/assets/143638502/082fb519-ea33-467b-ac14-4f9508a68790)
![スクリーンショット 2024-01-11 102143](https://github.com/sada-kaito/dreamer-reproduce/assets/143638502/242a1ced-0711-46ca-815f-21a40836ada9)  
import egl errorみたいなエラーがたまに出ます.  
  
**このエラーが出た時の対処法**  
僕はspyderで動かしていたので，その時の対処法を記しておきます．  
1.spyderを一度閉じるもしくはPCの再起動を行う．  
2.spyderを開き，test/t_dm_control.pyを開いて，一度実行する．  
3.その後に，dreamer_repro.pyを実行する．  

