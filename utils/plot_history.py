import matplotlib.pyplot as plt

def plot_training_history(history, path):
    '''
    Plots train loss, val loss, train accuracy and validation accuracy.
    Also, Saves the plot.

    Parameters:
    history: Containing training history 
    path: path to save the plot to

    Returns:
    None

    ''' 
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']

    # Get it for the validation-set 
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, linestyle='--', color='b', label='Training Loss')
    
    # Plot it for the val set.
    plt.plot(val_acc, linestyle='-', color='r', label='Test Acc.')
    plt.plot(val_loss, linestyle='--', color='r', label='Test Loss')
    
    # Setting y range
    plt.ylim((0,0.2))
    
    # Plot title and legend.
    plt.title('Training and Vaidation History')
    plt.legend()

    # Save the plot
    plt.savefig(path)