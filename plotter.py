import matplotlib.pyplot as plt
from IPython import display

plt.ion()
def plot(scores, meanScore):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Entrenando...')
    plt.xlabel('Numero de Juegos')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(meanScore)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(meanScore)-1, meanScore[-1], str(meanScore[-1]))
    plt.show(block=False)
    plt.pause(.1)