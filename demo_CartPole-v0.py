import numpy as np
import gym
from gym import wrappers
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from IPython import display
from JSAnimation.IPython_display import display_animation
from IPython.display import HTML


env = gym.make('CartPole-v0')  # シミュレータ環境の構築
frames = []
for episode in range(5):
    state = env.reset()  # エピソードを開始（環境の初期化）
    env.render()  # シミュレータ画面の出力
    screen = env.render(mode='rgb_array')  # notebook上での結果の可視化用
    frames.append(screen)
    done = False
    while not done:
        action = env.action_space.sample()  # 行動をランダムに選択
        next_state, reward, done, _ = env.step(action)  # 行動を実行し、次の状態、 報酬、 終端か否かの情報を取得
        env.render()
        screen = env.render(mode='rgb_array')
        frames.append(screen)
env.close()  # 画面出力の終了

# 結果の確認
plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
patch = plt.imshow(frames[0])
plt.axis('off')
  
def animate(i):
    patch.set_data(frames[i])
    
anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
HTML(anim.to_jshtml())