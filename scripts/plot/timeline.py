import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import textwrap
import datetime
import yaml

years = mdates.YearLocator()
months = mdates.MonthLocator()
years_fmt = mdates.DateFormatter("%Y")

with open("data/plots/timeline-citations.yaml", "r") as stream:
    try:
        data = yaml.safe_load(stream)

        data = data[:5]

        # Prepare data
        x = [d["pubDate"] for d in data]
        y = [d["numCitations"] for d in data]
        labels = [d["title"] for d in data]

        # Put data in correct format
        x = [datetime.datetime.strptime(d, "%Y-%m-%d") for d in x]

        # Set graph
        fig, ax = plt.subplots(figsize=(1, 1))

        p1 = ax.bar(x, y) #width=datetime.timedelta(3))

        # Log-scale because big stuff
        ax.set_yscale("log")

        # Set axis ticks
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_minor_locator(months)
        ax.xaxis.set_major_formatter(years_fmt)

        # Remove top and right axis
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Label each bar
        rects = ax.patches
        for rect, label in zip(rects, labels):
            label = textwrap.fill(label, width=30)
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom", fontsize=0.5
            )

        plt.bar(x, y)
        plt.savefig("plots/timeline.svg")
    except yaml.YAMLError as exc:
        raise(exc)