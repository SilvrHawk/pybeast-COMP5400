import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def extract_fitness_from_csv(file_path):
    """Extract fitness data from a CSV file with Generation, avg_fitness, max_fitness columns."""
    try:
        # Use pandas to read the CSV file which handles headers and parsing automatically
        df = pd.read_csv(file_path)

        # Extract columns from dataframe
        generations = df["Generation"].tolist()
        average_fitness = df["avg_fitness"].tolist()
        max_fitness = df["max_fitness"].tolist()

        return generations, average_fitness, max_fitness
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return [], [], []


def plot_fitness_data(generations, average_fitness, max_fitness):
    """Plot fitness data with trend lines."""
    plt.figure(figsize=(14, 7))

    # Plot raw data points with some transparency
    plt.scatter(
        generations,
        average_fitness,
        s=10,
        alpha=0.4,
        label="Average Fitness",
        color="blue",
    )
    plt.scatter(
        generations, max_fitness, s=10, alpha=0.4, label="Maximum Fitness", color="red"
    )

    # Create polynomial fit lines (degree 3 gives a good balance of fit without overfitting)
    degree = 3

    # Calculate polynomial coefficients
    avg_fit_coef = np.polyfit(generations, average_fitness, degree)
    max_fit_coef = np.polyfit(generations, max_fitness, degree)

    # Create polynomial functions
    avg_fit_poly = np.poly1d(avg_fit_coef)
    max_fit_poly = np.poly1d(max_fit_coef)

    # Generate smooth x values for plotting trend lines
    x_smooth = np.linspace(min(generations), max(generations), 1000)

    # Plot trend lines
    plt.plot(
        x_smooth, avg_fit_poly(x_smooth), "b-", linewidth=2.5, label="Avg Fitness Trend"
    )
    plt.plot(
        x_smooth, max_fit_poly(x_smooth), "r-", linewidth=2.5, label="Max Fitness Trend"
    )

    # # Add annotations showing overall trend direction
    # plt.annotate(
    #     f"Average trend: {'↑' if avg_fit_coef[0] > 0 else '↓'}",
    #     xy=(0.02, 0.95),
    #     xycoords="axes fraction",
    # )
    # plt.annotate(
    #     f"Maximum trend: {'↑' if max_fit_coef[0] > 0 else '↓'}",
    #     xy=(0.02, 0.90),
    #     xycoords="axes fraction",
    # )

    # Improve plot appearance
    plt.title("Evolution of Fitness over Generations", fontsize=16)
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Fitness Score", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="upper left")

    # Add average and max fitness values annotation
    plt.figtext(
        0.02,
        0.02,
        f"Final avg fitness: {average_fitness[-1]:.2f}\nFinal max fitness: {max_fitness[-1]:.2f}",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    # plt.savefig("fitness_evolution.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    csv_file_path = "other/log.csv"
    generations, avg_fitness, max_fitness = extract_fitness_from_csv(csv_file_path)

    plot_fitness_data(generations, avg_fitness, max_fitness)
