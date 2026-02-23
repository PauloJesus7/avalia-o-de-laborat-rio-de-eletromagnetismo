import matplotlib
matplotlib.use('TkAgg') # Força a abrir uma janela do Windows estável, livre do VS Code

import numpy as np
import matplotlib.pyplot as plt
import time

# Parâmetros do Guia WR-75
largura = 19.05e-3  # 19.05 mm
comprimento_z = 0.15 # 15 cm
frequencia = 12e9   # 12 GHz
omega = 2 * np.pi * frequencia

mu = 4 * np.pi * 1e-7
epsilon = 8.854 * 1e-12

# Resolução da malha
dx = largura / 50       
dz = dx           
Nx = int(largura / dx)  
Nz = int(comprimento_z / dz)  

# Estabilidade do tempo
dt = 0.99 / ((1 / np.sqrt(mu * epsilon)) * np.sqrt((1/dx)**2 + (1/dz)**2))

Ey = np.zeros((Nx, Nz))
Hx = np.zeros((Nx, Nz))
Hz = np.zeros((Nx, Nz))

# O Molde do Modo Dominante TE10
eixo_x = np.linspace(0, largura, Nx)
perfil_TE10 = np.sin(np.pi * eixo_x / largura)

# --- CONFIGURAÇÃO VISUAL BLINDADA ---
plt.ion() # Liga o modo interativo
fig, ax = plt.subplots(figsize=(8, 6))

# aspect='auto' corrige o gráfico "macarrão", esticando para preencher a tela
cax = ax.imshow(Ey.T, cmap='RdBu_r', vmin=-1.2, vmax=1.2, origin='lower', 
                extent=[0, largura*1000, 0, comprimento_z*1000], aspect='auto')

ax.set_title("Formação e Propagação do Modo Dominante TE10 (12 GHz)", fontsize=14, pad=15)
ax.set_xlabel("Largura do Guia - Eixo X (mm)", fontsize=12)
ax.set_ylabel("Direção de Propagação - Eixo Z (mm)", fontsize=12)

# Legenda arrumada
ax.axhline(y=1, color='black', linestyle='--', linewidth=1.5, label='Plano de Excitação (Fonte)')
ax.legend(loc='upper right', framealpha=0.9)

fig.colorbar(cax, label="Intensidade do Campo Elétrico (Ey)")
plt.tight_layout()

# FORÇA A JANELA A APARECER ANTES DO LOOP COMEÇAR
plt.show(block=False) 

print("Calculando FDTD e gerando animação...")

# Loop de simulação no tempo
for step in range(600): # Aumentei para 600 frames para dar tempo de ver bem
    tempo_atual = step * dt
    
    # Maxwell (Faraday e Ampère)
    Hx[:, :-1] += (dt / (mu * dz)) * (Ey[:, 1:] - Ey[:, :-1])
    Hz[:-1, :] -= (dt / (mu * dx)) * (Ey[1:, :] - Ey[:-1, :])
    Ey[1:-1, 1:-1] += (dt / epsilon) * ( 
        (Hx[1:-1, 1:-1] - Hx[1:-1, :-2]) / dz - 
        (Hz[1:-1, 1:-1] - Hz[:-2, 1:-1]) / dx 
    )
    
    # Paredes PEC (Branco na animação)
    Ey[0, :] = 0    
    Ey[-1, :] = 0   
    
    # Injetando o modo TE10 harmonicamente
    Ey[:, 1] = perfil_TE10 * np.sin(omega * tempo_atual)

    # Atualiza o frame de forma segura (sem travar a interface)
    if step % 6 == 0:
        cax.set_data(Ey.T)
        fig.canvas.draw()         # Força o desenho
        fig.canvas.flush_events() # Libera a interface para não congelar
        time.sleep(0.01)          # Dá um respiro para o processador

print("Simulação concluída!")
plt.ioff()
plt.show() # Mantém a janela aberta no final