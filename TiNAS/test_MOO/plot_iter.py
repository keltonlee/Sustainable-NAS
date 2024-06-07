import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

with open('../vanila_iter_info.json', 'r') as f:
    data = json.load(f)

fig, ax = plt.subplots()

colors = cm.rainbow(np.linspace(0, 1, len(data)))
print(len(data), len(data[0]))

for i, iteration in enumerate(data):
    accuracies = [model['acc'] for model in iteration]
    carbons = [model['cf'] for model in iteration]
    
    ax.scatter(accuracies, carbons, color=colors[i], label=f'Iteration {i+1}')

ax.set_title('Accuracy vs. Carbon Footprint')
ax.set_xlabel('Accuracy')
ax.set_ylabel('Carbon Footprint')

ax.legend()

plt.show()