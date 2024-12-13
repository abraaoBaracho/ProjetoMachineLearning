import numpy as np
import gym
import random
from IPython.display import clear_output
import matplotlib.pyplot as plt
from time import sleep
import pickle

# Corrigir compatibilidade com numpy.bool8
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

def run (episodios, treinamento=True, render=False):
    
    # Inicializar o ambiente
    env = gym.make('Taxi-v3', render_mode='human' if render else None)
    
    if(treinamento):
        # Inicializar a tabela Q com zeros
        tabela_q = np.zeros([env.observation_space.n, env.action_space.n])
    else:
        f = open('taxi.pkl', 'rb')
        tabela_q = pickle.load(f)
        f.close()

    # Hiperparâmetros
    alpha = 0.9
    gamma = 0.9
    epsilon = 1
    epsilon_decay_rate = 0.0001        # taxa de decaimento epsilon. 1/0.0001 = 10,000
    rng = np.random.default_rng()   # gerador de números aleatórios
    
    recompensa_por_episodio = np.zeros(episodios)

    # Treinamento
    for i in range(episodios):
        estado = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        truncated = False       # Verdadeiro quando ações > 200
        terminado = False

        recompensa = 0
        
        while(not terminado and not truncated):
            # Decidir entre explorar e explorar
            if treinamento and rng.random() < epsilon:
                acao = env.action_space.sample()
            else:
                acao = np.argmax(tabela_q[estado,:])

            # Executar a ação no ambiente
            proximo_estado, recompensa, terminado, truncated, _ = env.step(acao)

            recompensa += recompensa
            
            # Atualizar a tabela Q
            if treinamento:
                tabela_q[estado,acao] = tabela_q[estado,acao] + alpha * (
                    recompensa + gamma * np.max(tabela_q[proximo_estado,:]) - tabela_q[estado, acao]
                )
                
            estado = proximo_estado
            
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        
        if(epsilon==0):
                alpha = 0.0001
                

        recompensa_por_episodio[i] = recompensa
        
    env.close()
    
    sum_rewards = np.zeros(episodios)
    for t in range(episodios):
        sum_rewards[t] = np.sum(recompensa_por_episodio[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('taxi.png')

    if treinamento:
        f = open("taxi.pkl","wb")
        pickle.dump(tabela_q, f)
        f.close()
        
    
    plt.plot(sum_rewards)
    plt.title('Desempenho do Agente ao Longo dos Episódios')
    plt.xlabel('Episódios')
    plt.ylabel('Soma das Recompensas (100 Episódios)')
    plt.savefig('taxi.png')
    
    print("Treinamento terminado.\n")

if __name__ == '__main__':
    run(15000)

    run(10, treinamento=False, render=True)
    
    plt.show()
