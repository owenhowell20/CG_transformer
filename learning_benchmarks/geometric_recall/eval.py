import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Assume you have a function `load_model` that loads your pretrained model
# model = load_model()


# Evaluate a single sample from the dataset
def evaluate_model(model, dataset, idx=0):
    # Get the sample data
    sample = dataset[idx]
    question = sample["question"]  # (3,)
    true_answer = sample["answer"]  # (3,)

    # Get the model prediction
    # Assuming your model takes in a query and outputs a predicted vector
    with torch.no_grad():
        predicted_answer = model(sample["query"])  # Predicted answer (3,)

    # Normalize the predicted answer (since it's supposed to be a unit vector)
    predicted_answer = predicted_answer / predicted_answer.norm(dim=-1, keepdim=True)

    return question, true_answer, predicted_answer


# Function to plot vectors in 3D
def plot_vectors(true_vector, predicted_vector, question_vector=None):
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot true answer vector
    ax.quiver(
        0,
        0,
        0,
        true_vector[0],
        true_vector[1],
        true_vector[2],
        color="r",
        label="True Answer",
    )

    # Plot predicted answer vector
    ax.quiver(
        0,
        0,
        0,
        predicted_vector[0],
        predicted_vector[1],
        predicted_vector[2],
        color="b",
        label="Predicted Answer",
    )

    if question_vector is not None:
        ax.quiver(
            0,
            0,
            0,
            question_vector[0],
            question_vector[1],
            question_vector[2],
            color="g",
            label="Question Vector",
        )

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set limits to ensure all vectors fit
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Add legend
    ax.legend()

    # Show plot
    plt.show()


# Evaluate and visualize the performance
def evaluate_and_visualize(model, dataset, idx=0):
    question, true_answer, predicted_answer = evaluate_model(model, dataset, idx)

    # Visualize
    plot_vectors(true_answer, predicted_answer, question)


if __name__ == "__main__":
    model = 1
    dataset = 1

    evaluate_and_visualize(model, dataset, idx=5)
