import sys
import json
import matplotlib.pyplot as plt

try:
    with open('history.json') as f:
        history = json.load(f)
except FileNotFoundError:
    print("Error: 'history.json' not found. Run train.py first.")
    sys.exit(1)

plt.figure()
plt.plot(history['loss'], label='training loss')
plt.plot(history['val_loss'], label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Learning Curves')
plt.legend()
plt.savefig('loss_curve.png')

plt.figure()
plt.plot(history['accuracy'], label='training acc')
plt.plot(history['val_accuracy'], label='validation acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Learning Curves')
plt.legend()
plt.savefig('accuracy_curve.png')

plt.show()
