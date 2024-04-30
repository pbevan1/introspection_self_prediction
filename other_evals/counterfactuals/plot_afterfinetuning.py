import plotly.graph_objs as go

# Data
models = ["Ask the model what option it would give instead"]

accuracy_before = [0.586]  # Macro Accuracy
accuracy_after = [0.595]  # Macro Accuracy
accuracy_after_finetuning_same_task = [0.758]
ci_accuracy_before = [0.04, 0.04]  # CI for Accuracy
ci_accuracy_after = [0.04, 0.04]  # CI for Accuracy

# Colors for the bars

# Define the figure
fig = go.Figure()

# Define bar for Ground Truth - Affected
# fig.add_trace(go.Bar(
#     name='Ground Truth - Affected',
#     x=models,
#     y=affected,
#     text=['{:.1%}'.format(val) for val in affected],
#     error_y=dict(type='data', array=ci_affected, visible=True),
#     marker_color='blue'
# ))

# # Define bar for Ground Truth - Unaffected
# fig.add_trace(go.Bar(
#     name='Ground Truth - Unaffected',
#     x=models,
#     y=unaffected,
#     text=['{:.1%}'.format(val) for val in unaffected],
#     error_y=dict(type='data', array=ci_unaffected, visible=True),
#     marker_color='orange'
# ))

fig.add_trace(
    go.Bar(
        name="Before finetuning<br>Accuracy<br>(Balanced over two ground truth classes)",
        x=models,
        y=accuracy_before,
        text=["{:.1%}".format(val) for val in accuracy_before],
        error_y=dict(type="data", array=ci_accuracy_before, visible=True),
        marker_color="purple",
    )
)

# Define bar for Average Accuracy
fig.add_trace(
    go.Bar(
        name="Finetuning on other introspective tasks<br>9EXL6W9A",
        x=models,
        y=accuracy_after,
        text=["{:.1%}".format(val) for val in accuracy_after],
        error_y=dict(type="data", array=ci_accuracy_after, visible=True),
        marker_color="orange",
    )
)

# Define bar for Average Accuracy
fig.add_trace(
    go.Bar(
        name="Finetuning on this task, on bbh, test on mmlu<br>9FgW32xp",
        x=models,
        y=accuracy_after_finetuning_same_task,
        text=["{:.1%}".format(val) for val in accuracy_after_finetuning_same_task],
        error_y=dict(type="data", array=ci_accuracy_after, visible=True),
        marker_color="red",
    )
)


# Set the layout for the figure
fig.update_layout(
    title="Accuracy ",
    xaxis=dict(title="Model"),
    yaxis=dict(title="Score", range=[0, 1.0]),
    # legend outside
    # legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    barmode="group",
)

# Show figure
fig.show()
