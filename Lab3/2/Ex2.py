import numpy as np
import matplotlib.pyplot as plt

#Fig 1
fig1, axs1 = plt.subplots(2, 1)

# Fig 1A
x = np.linspace(0, 1, 1000)
y = np.sin(14 * np.pi * x + np.pi / 4)

axs1[0].scatter(x, y, c=np.abs(y), cmap='Greens_r', s=5)
axs1[0].set_xlabel('Timp (Esantioane)')
axs1[0].set_ylabel('Amplitudine')
axs1[0].axhline(0, color='black', linewidth=1)

sample_idx = 500
axs1[0].plot([x[sample_idx], x[sample_idx]], [0, y[sample_idx]], color='red')
axs1[0].scatter(x[sample_idx], y[sample_idx], color='red')

# Fig 1B
omega = 1
z = np.exp(1j * -2 * np.pi * omega * x) * y

axs1[1].scatter(np.real(z), np.imag(z), c=np.abs(z), cmap='Blues_r', s=5) 
axs1[1].set_xlabel('Real')
axs1[1].set_ylabel('Imaginar')
axs1[1].axhline(0, color='black', linewidth=1)
axs1[1].axvline(0, color='black', linewidth=1)
axs1[1].axis('equal')

sample_point = z[sample_idx] 
axs1[1].plot([0, np.real(sample_point)], [0, np.imag(sample_point)], color='red')
axs1[1].scatter(np.real(sample_point), np.imag(sample_point), color='red', zorder=5)

# Show Fig 1
plt.tight_layout()
plt.savefig('Fig1.pdf')
plt.savefig('Fig1.png')
plt.show()

# Fig 2
fig2, axs2 = plt.subplots(2, 2)

i = 0
for omega in [1, 2, 5, 7]:
    
    z = np.exp(1j * -2 * np.pi * omega * x) * y
    ax = axs2.flatten()[i]
    i += 1
    
    ax.set_title(f'Omega = {omega}')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginar')
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.axis('equal')
    
    # Without animation
    # ax.scatter(np.real(z), np.imag(z), c=np.abs(z), cmap='inferno', s=5) 
    
    # With animation
    anim = 50
    for idx in range(anim, len(x)+1, anim):
        ax.scatter(np.real(z[:idx]), np.imag(z[:idx]), c=np.abs(z[:idx]), cmap='inferno', s=5)
        plt.pause(0.00001)
        
# Show Fig 2
plt.tight_layout()
plt.savefig('Fig2.pdf')
plt.savefig('Fig2.png')
plt.show()
