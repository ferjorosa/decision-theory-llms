import gradio as gr
import pyagrum as gum
import pyagrum.lib.image as gumimage
import tempfile

def show_inference(_=None):
    # Create an influence diagram
    infdiag = gum.fastID("Chance->*Decision1->Chance2->$Utility<-Chance3<-*Decision2<-Chance->Utility")
    
    # Set up the inference engine
    ie = gum.ShaferShenoyLIMIDInference(infdiag)
    
    # Add the no-forgetting assumption for classical influence diagrams
    ie.addNoForgettingAssumption(["Decision1", "Decision2"])
    
    # Perform the inference
    ie.makeInference()
    
    # Create a temporary file for the image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img_path = tmp.name
    
    # Export the inference result to a PNG file
    gumimage.exportInference(infdiag, img_path, engine=ie)
    
    return img_path

# Create a simple Gradio interface with just a button and image output
demo = gr.Interface(
    fn=show_inference,
    inputs="button",
    outputs=gr.Image(label="Inference Result"),
    title="PyAgrum Inference Visualization"
)

if __name__ == "__main__":
    demo.launch()