 import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cmath 
import time

# =======================================================
# CLASSE PARA O MODO TE (Transversal Elétrico)
# =======================================================
class Modo_TEmn():
    def __init__(self, largura = 19.05, # WR-75 (em mm)
                 altura = 9.53,         # WR-75 (em mm)
                 frequencia = 12*10**9, # 12 GHz
                 permissividade_relativa = 1, 
                 permeabilidade_relativa = 1, 
                 plano = 'xy'):
        
        self.plano = plano
        self.pi = np.pi
        self.A = 1 # Amplitude
        self.frequencia = frequencia

        # Modo dominante TE10
        self.m = 1
        self.n = 0

        # Dimensões geométricas (convertidas para metros)
        self.largura = largura / 1000 
        self.altura = altura / 1000 
        self.profundidade = 0.05 

        # Constantes Físicas
        self.light_speed = 299792458 
        self.mu_0 = 4 * self.pi * 10**-7 
        self.epsilon_0 = 8.854 * 10**-12 
        
        # Propriedades do Meio
        self.mu = permeabilidade_relativa * self.mu_0 
        self.epsilon = permissividade_relativa * self.epsilon_0
        
        self.pontos_por_dimensao = 50

        # Cálculos Teóricos
        self.omega = self.calcula_omega() 
        self.k = self.calcula_k() 
        self.k_c = self.calcula_k_c() 
        self.beta = self.calcula_beta() 

        # Configurações de malha e plotagem
        self.escolha_plano()
        self.cos_mx = self.cosseno_x()
        self.cos_ny = self.cosseno_y()
        self.sen_mx = self.seno_x()
        self.sen_ny = self.seno_y()
        self.expz = self.exp_z()

    def calcula_omega(self): return self.frequencia * 2 * self.pi
    def calcula_k(self): return self.omega * np.sqrt(self.mu * self.epsilon)
    def calcula_k_c(self): return np.sqrt((self.m * self.pi / self.largura)**2 + (self.n * self.pi / self.altura)**2)
    def calcula_beta(self): return cmath.sqrt(self.k**2 - self.k_c**2)

    def criar_meshgrid_xy(self):
        x = np.linspace(0, self.largura, self.pontos_por_dimensao)
        y = np.linspace(0, self.altura, self.pontos_por_dimensao)
        X, Y = np.meshgrid(x, y, indexing='ij')
        return X, Y, np.ones_like(X)

    def criar_meshgrid_xz(self):
        x = np.linspace(0, self.largura, self.pontos_por_dimensao)
        z = np.linspace(0, self.profundidade, self.pontos_por_dimensao)
        X, Z = np.meshgrid(x, z, indexing='ij')
        return X, np.ones_like(X), Z

    def criar_meshgrid_yz(self):
        y = np.linspace(0, self.altura, self.pontos_por_dimensao)
        z = np.linspace(0, self.profundidade, self.pontos_por_dimensao)
        Y, Z = np.meshgrid(y, z, indexing='ij')
        return np.ones_like(Y), Y, Z

    def escolha_plano(self):
        if self.plano == 'xy': self.x, self.y, self.z = self.criar_meshgrid_xy()
        elif self.plano == 'xz': self.x, self.y, self.z = self.criar_meshgrid_xz()
        elif self.plano == 'yz': self.x, self.y, self.z = self.criar_meshgrid_yz()

    def cosseno_x(self): return np.cos(self.m*self.pi*self.x/self.largura)
    def cosseno_y(self): return np.cos(self.n*self.pi*self.y/self.altura)
    def seno_x(self): return np.sin(self.m*self.pi*self.x/self.largura)
    def seno_y(self): return np.sin(self.n*self.pi*self.y/self.altura)
    def exp_z(self): return np.exp(-1j*self.beta*self.z)

    def H_z(self): return self.A * self.cos_mx * self.cos_ny * self.expz
    def H_x(self): return (1j*self.beta*self.m*self.pi/(self.k_c**2*self.largura)) * self.A * self.sen_mx * self.cos_ny * self.expz
    def H_y(self): return (1j*self.beta*self.n*self.pi/(self.k_c**2*self.altura)) * self.A * self.cos_mx * self.sen_ny * self.expz
    def E_x(self): return (1j*self.omega*self.mu*self.n*self.pi/(self.k_c**2 *self.altura)) * self.A * self.cos_mx * self.cos_ny * self.expz
    def E_y(self): return (-1j*self.omega*self.mu*self.m*self.pi/(self.k_c**2 *self.largura)) * self.A * self.sen_mx * self.cos_ny * self.expz

    def calcula_campos(self):
        self.Hx = np.real(self.H_x())
        self.Hy = np.real(self.H_y())
        self.Hz = np.real(self.H_z())
        self.Ex = np.real(self.E_x())
        self.Ey = np.real(self.E_y())
        self.Ez = np.zeros_like(self.Ex)

    def plota_campo_vetorial(self, campo = 'magnetico'):
        if self.plano == 'xy': abscissas, ordenadas, u, v = self.x, self.y, self.Hx if campo=='magnetico' else self.Ex, self.Hy if campo=='magnetico' else self.Ey
        elif self.plano == 'xz': abscissas, ordenadas, u, v = self.x, self.z, self.Hx if campo=='magnetico' else self.Ex, self.Hz if campo=='magnetico' else self.Ez
        elif self.plano == 'yz': abscissas, ordenadas, u, v = self.z, self.y, self.Hz if campo=='magnetico' else self.Ez, self.Hy if campo=='magnetico' else self.Ey

        plt.figure()
        plt.quiver(abscissas, ordenadas, u, v, color = 'blue')
        plt.title(f'Campo {campo.capitalize()} (TE10) no plano {self.plano}')
        plt.grid()
        plt.show()

    def plot3DField(self, campo = 'magnetico', componente = 'x'):
        imagem = getattr(self, f"{'H' if campo == 'magnetico' else 'E'}{componente}")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ground = np.min(imagem)
        ax.contourf(self.x, self.y, imagem, zdir='z', offset=ground, cmap='Spectral')
        ax.plot_surface(self.x, self.y, imagem, cmap='Spectral', alpha=0.7)
        ax.set_xlabel('Eixo X (m)'); ax.set_ylabel('Eixo Y (m)'); ax.set_zlabel(f'Amplitude')
        ax.set_title(f'Campo {campo.capitalize()} ({componente}) - Modo TE10')
        plt.show()

    # =========================================================================
    # NOVA FUNÇÃO: SIMULADOR NUMÉRICO FDTD (Célula de Yee 2D)
    # =========================================================================
    def simular_fdtd(self, comprimento_z=0.1, passos_tempo=350):
        print("Iniciando simulação FDTD (Propagação no Tempo)...")
        
        # Resolução da malha espacial
        dx = self.largura / 40       
        dz = dx           

        Nx = int(self.largura / dx)  
        Nz = int(comprimento_z / dz)  

        # Condição de Courant para estabilidade no tempo
        velocidade_onda = 1 / np.sqrt(self.mu * self.epsilon)
        dt = 0.99 / (velocidade_onda * np.sqrt((1/dx)**2 + (1/dz)**2))

        # Inicialização das matrizes de campo com zeros
        Ey_fdtd = np.zeros((Nx, Nz))
        Hx_fdtd = np.zeros((Nx, Nz))
        Hz_fdtd = np.zeros((Nx, Nz))

        # Perfil espacial transversal (Modo TE10 é um meio-seno no eixo X)
        eixo_x = np.linspace(0, self.largura, Nx)
        perfil_TE10 = np.sin(self.pi * eixo_x / self.largura)

        # Configuração da janela gráfica iterativa
        plt.ion() 
        fig, ax = plt.subplots(figsize=(8, 5))
        cax = ax.imshow(Ey_fdtd.T, cmap='seismic', vmin=-1, vmax=1, origin='lower', 
                        extent=[0, self.largura*1000, 0, comprimento_z*1000])
        ax.set_title(f"FDTD: Propagação do Modo TE10 ({self.frequencia/1e9} GHz)")
        ax.set_xlabel("Largura X (mm)")
        ax.set_ylabel("Comprimento de Propagação Z (mm)")
        fig.colorbar(cax, label="Amplitude do Campo Elétrico (Ey)")

        # Loop no tempo (Equações de Maxwell discretizadas)
        for step in range(passos_tempo):
            tempo_atual = step * dt
            
            # 1. Lei de Faraday (Atualiza Campo Magnético)
            Hx_fdtd[:, :-1] += (dt / (self.mu * dz)) * (Ey_fdtd[:, 1:] - Ey_fdtd[:, :-1])
            Hz_fdtd[:-1, :] -= (dt / (self.mu * dx)) * (Ey_fdtd[1:, :] - Ey_fdtd[:-1, :])
            
            # 2. Lei de Ampère (Atualiza Campo Elétrico)
            Ey_fdtd[1:-1, 1:-1] += (dt / self.epsilon) * ( 
                (Hx_fdtd[1:-1, 1:-1] - Hx_fdtd[1:-1, :-2]) / dz - 
                (Hz_fdtd[1:-1, 1:-1] - Hz_fdtd[:-2, 1:-1]) / dx 
            )
            
            # 3. Condições de Contorno (Paredes de Metal PEC em x=0 e x=a)
            Ey_fdtd[0, :] = 0    
            Ey_fdtd[-1, :] = 0   
            
            # 4. Excitação na entrada do guia (z=1) Injetando o modo
            Ey_fdtd[:, 1] = perfil_TE10 * np.sin(self.omega * tempo_atual)

            # 5. Atualiza o gráfico na tela a cada 5 iterações
            if step % 5 == 0:
                cax.set_data(Ey_fdtd.T)
                plt.pause(0.001)

        plt.ioff()
        print("Simulação FDTD concluída com sucesso!")
        plt.show()


# =======================================================
# CLASSE PARA O MODO TM (Transversal Magnético)
# =======================================================
class Modo_TMmn():
    def __init__(self, largura = 19.05, 
                 altura = 9.53,         
                 frequencia = 12*10**9, 
                 permissividade_relativa = 1, 
                 permeabilidade_relativa = 1, 
                 plano = 'xy'):
        
        self.plano = plano
        self.pi = np.pi
        self.B = 1 
        self.frequencia = frequencia

        # Modo fundamental TM11
        self.m = 1
        self.n = 1

        self.largura = largura/1000
        self.altura = altura/1000
        self.profundidade = 0.05 

        self.light_speed = 299792458 
        self.mu_0 = 4 * self.pi * 10**-7 
        self.epsilon_0 = 8.854 * 10**-12 
        
        self.mu = permeabilidade_relativa * self.mu_0 
        self.epsilon = permissividade_relativa * self.epsilon_0
        
        self.pontos_por_dimensao = 50 

        self.omega = self.calcula_omega() 
        self.k = self.calcula_k() 
        self.k_c = self.calcula_k_c() 
        self.beta = self.calcula_beta()

        self.escolha_plano()
        self.cos_mx = self.cosseno_x()
        self.cos_ny = self.cosseno_y()
        self.sen_mx = self.seno_x()
        self.sen_ny = self.seno_y()
        self.expz = self.exp_z()

    def calcula_omega(self): return self.frequencia * 2 * self.pi
    def calcula_k(self): return self.omega * np.sqrt(self.mu * self.epsilon)
    def calcula_k_c(self): return np.sqrt((self.m * self.pi / self.largura)**2 + (self.n * self.pi / self.altura)**2)
    def calcula_beta(self): return cmath.sqrt(self.k**2 - self.k_c**2)

    def criar_meshgrid_xy(self):
        x = np.linspace(0, self.largura, self.pontos_por_dimensao)
        y = np.linspace(0, self.altura, self.pontos_por_dimensao)
        X, Y = np.meshgrid(x, y, indexing='ij')
        return X, Y, np.ones_like(X)

    def criar_meshgrid_xz(self):
        x = np.linspace(0, self.largura, self.pontos_por_dimensao)
        z = np.linspace(0, self.profundidade, self.pontos_por_dimensao)
        X, Z = np.meshgrid(x, z, indexing='ij')
        return X, np.ones_like(X), Z

    def criar_meshgrid_yz(self):
        y = np.linspace(0, self.altura, self.pontos_por_dimensao)
        z = np.linspace(0, self.profundidade, self.pontos_por_dimensao)
        Y, Z = np.meshgrid(y, z, indexing='ij')
        return np.ones_like(Y), Y, Z

    def escolha_plano(self):
        if self.plano == 'xy': self.x, self.y, self.z = self.criar_meshgrid_xy()
        elif self.plano == 'xz': self.x, self.y, self.z = self.criar_meshgrid_xz()
        elif self.plano == 'yz': self.x, self.y, self.z = self.criar_meshgrid_yz()

    def cosseno_x(self): return np.cos(self.m*self.pi*self.x/self.largura)
    def cosseno_y(self): return np.cos(self.n*self.pi*self.y/self.altura)
    def seno_x(self): return np.sin(self.m*self.pi*self.x/self.largura)
    def seno_y(self): return np.sin(self.n*self.pi*self.y/self.altura)
    def exp_z(self): return np.exp(-1j*self.beta*self.z)

    def E_z(self): return self.B * self.sen_mx * self.sen_ny * self.expz
    def E_x(self): return (-1j * self.beta * self.m * self.pi / (self.largura * self.k_c**2)) * self.B * self.cos_mx * self.sen_ny * self.expz
    def E_y(self): return (-1j * self.beta * self.n * self.pi / (self.altura * self.k_c**2)) * self.B * self.sen_mx * self.cos_ny * self.expz
    def H_x(self): return (1j * self.omega * self.epsilon * self.n * self.pi / (self.altura * self.k_c**2)) * self.B * self.sen_mx * self.cos_ny * self.expz
    def H_y(self): return (-1j * self.omega * self.epsilon * self.m * self.pi / (self.largura * self.k_c**2)) * self.B * self.cos_mx * self.sen_ny * self.expz

    def calcula_campos(self):
        self.Ex = np.real(self.E_x())
        self.Ey = np.real(self.E_y())
        self.Ez = np.real(self.E_z())
        self.Hx = np.real(self.H_x())
        self.Hy = np.real(self.H_y())
        self.Hz = np.zeros_like(self.Hx)

    def plota_campo_vetorial(self, campo = 'magnetico'):
        if self.plano == 'xy': abscissas, ordenadas, u, v = self.x, self.y, self.Hx if campo=='magnetico' else self.Ex, self.Hy if campo=='magnetico' else self.Ey
        elif self.plano == 'xz': abscissas, ordenadas, u, v = self.x, self.z, self.Hx if campo=='magnetico' else self.Ex, self.Hz if campo=='magnetico' else self.Ez
        elif self.plano == 'yz': abscissas, ordenadas, u, v = self.z, self.y, self.Hz if campo=='magnetico' else self.Ez, self.Hy if campo=='magnetico' else self.Ey

        plt.figure()
        plt.quiver(abscissas, ordenadas, u, v, color = 'blue')
        plt.title(f'Campo {campo.capitalize()} (TM11) no plano {self.plano}')
        plt.grid()
        plt.show()

    def plot3DField(self, campo = 'magnetico', componente = 'x'):
        imagem = getattr(self, f"{'H' if campo == 'magnetico' else 'E'}{componente}")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ground = np.min(imagem)
        ax.contourf(self.x, self.y, imagem, zdir='z', offset=ground, cmap='Spectral')
        ax.plot_surface(self.x, self.y, imagem, cmap='Spectral', alpha=0.7)
        ax.set_xlabel('Eixo X (m)'); ax.set_ylabel('Eixo Y (m)'); ax.set_zlabel(f'Amplitude')
        ax.set_title(f'Campo {campo.capitalize()} ({componente}) - Modo TM11')
        plt.show()


# =======================================================
# ÁREA DE TESTE E EXECUÇÃO DO USUÁRIO
# =======================================================
if __name__ == "__main__":
    
    print("==============================================")
    print("SIMULADOR DE GUIA DE ONDAS RETANGULAR (WR-75)")
    print("==============================================\n")
    
    modo_te = Modo_TEmn(plano='xy') 
    
    # --- 1. Simulando FDTD (Animação Numérica no Tempo) ---
    # Isso vai abrir a janela da animação. Assista até o fim!
    modo_te.simular_fdtd(comprimento_z=0.1, passos_tempo=350)
    
    # Limpa a memória gráfica do seu computador para não dar o erro do Qt
    plt.close('all') 
    
    # --- 2. Resultados Analíticos Estacionários ---
    print("\nCalculando os Gráficos 3D estacionários...")
    modo_te.calcula_campos()
    
    # Vai abrir o gráfico 3D estático (Você precisa fechar a janela dele para o programa encerrar sozinho)
    modo_te.plot3DField(campo='magnetico', componente='z')
    
    print("Programa encerrado com sucesso!")