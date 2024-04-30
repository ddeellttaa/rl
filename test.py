import gym
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
import numpy as np
'''
python reforcement learning 
'''
class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DQN,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
     
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
          
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
 
       
        

    def forward(self,x):
        return self.model(x)
    
class DQNAgent(object):
    
    def __init__(self):
        self.env = gym.make("LunarLander-v2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    def solve(self):
        print(self.device)
        in_dim = self.env.observation_space.shape[0]
        out_dim = self.env.action_space.n
        gamma = 0.99
        epsilon = 0.25
        
        losses = []
        rewards = [] 
        avg_rewards = []       

        model = DQN(in_dim, out_dim).to(self.device)
       # model =  torch.load("data/model1999.pth", map_location = self.device)
        model.train()
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.7)    
        state =  self.env.reset()
        state = torch.from_numpy(state).float().to(self.device)

        epoch = 2000
        epochs = [] 
        for i in range(epoch):
            # print(i)
            self.env.reset()
            done = 0
            R=0
            J=0
            C=0
            l = 0

            model.train()
            while done != 1:
                C+=1
                random = torch.rand(1).item()
                model.eval()
                with torch.no_grad():
                    action = torch.argmax(model(state)) if random > epsilon else torch.randint(high=self.env.action_space.n, size=(1,))[0].item()
                model.train()
                action = int(action)
                next_state, reward, done, _ = self.env.step(action)
                R+=reward
                next_state = torch.from_numpy(next_state).float().to(self.device)
                next_acton = torch.argmax(model(next_state))
                #reward = torch.tensor([reward], device=self.device)
                q_value = model(state)[action]
                e_q_value = reward + gamma * model(next_state)[next_acton].detach()*(1 - done)
              
                l = loss(q_value,e_q_value)
                J+=int(l)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                state = next_state
                
                #self.env.render()
            
                # if (i+1) % 100 == 0:
                #     self.env.render()
            losses.append(J/C)
            rewards.append(R)
            avg_rewards.append(np.mean(rewards[-100:]))
            epochs.append(i)
            print(f"epoch:{i}  rewards:{R} avg_reward:{np.mean(rewards[-100:])} time:{C}")
      
            if (i+1) % 200 == 0:
                torch.save(model, f'model{i}.pth')
            # if R>200:
            #     torch.save(model, f'good_model{i}.pth')
            self.env.close()
            scheduler.step()
            if (i + 1) % 100 ==0:
                plt.figure(figsize=(24,6))
                plt.plot(epochs,losses,label="loss")
                plt.xlabel("epoch")
                plt.legend()
                plt.savefig(f"model{i}loss.png", dpi = 400)
                plt.close() 
                plt.figure(figsize=(24,6))
                plt.plot(epochs,rewards,label="reward")
                plt.plot(epochs,avg_rewards,label = "avg_reward")
                plt.xlabel("epoch")
                plt.legend()
                plt.savefig(f"model{i}reward.png", dpi = 400)
                plt.close()
        return l     
    
        
if __name__ == "__main__":
    agent = DQNAgent() 
    agent.solve()




