import gradio as gr

# Define the update function
def update_row(choice):
    if choice == "Option 1":
        return gr.update(visible=True, label="Updated Textbox 1")
    elif choice == "Option 2":
        return gr.update(visible=True, label="Updated Textbox 2")
    else:
        return gr.update(visible=False)

# Define the Gradio components
dropdown = gr.Dropdown(choices=["Option 1", "Option 2", "Hide"], label="Choose an Option")
textbox1 = gr.Textbox(label="Textbox 1")
textbox2 = gr.Textbox(label="Textbox 2")

# Use the `change` event to update the textboxes
dropdown.change(fn=update_row, inputs=dropdown, outputs=[textbox1, textbox2])

# Launch the interface
demo = gr.Interface(fn=lambda x: x, inputs=[dropdown, textbox1, textbox2], outputs="text")
demo.launch()
