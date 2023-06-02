import os
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
    metrics_file = 'checkpoints/run46/metrics.json'
    if not os.path.exists(metrics_file):
        raise FileNotFoundError('UwU')
    with open(metrics_file, 'r') as f:
        obj = json.load(f)
        
    gen_loss = obj['generator']['loss']
    disc_loss = obj['discriminator']['loss']
    
    plt.plot(gen_loss, label='Generator')
    plt.plot(disc_loss, label='Discriminator')
    plt.suptitle('Loss over training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.savefig('loss.png')
    plt.show()        
    print(f'Exit')