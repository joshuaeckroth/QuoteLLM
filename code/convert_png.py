from PIL import Image
import glob

def convert_to_150_dpi(input_path, output_path):
    # Open the PNG image
    img = Image.open(input_path)

    # Set the resolution to 150 dpi
    img = img.convert('RGB')  # Ensure the image is in RGB mode
    img = img.resize((int(img.width * 150 / img.info['dpi'][0]), int(img.height * 150 / img.info['dpi'][1])), Image.ANTIALIAS)
    img.info['dpi'] = (150, 150)  # Set the DPI to 150

    # Save the result
    img.save(output_path, dpi=(150, 150))


def main():
    for image_path in glob.glob('/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/*/*'):
        image_name = image_path.split('/')[-1]
        if (image_name.split('.')[-1] != 'tex'):
            print(image_name)
            output_path = '/Users/skyler/Desktop/QuoteLLM/all-models-results/dpi_visualization/' + image_name
            convert_to_150_dpi(image_path, output_path)
        else:
            print()
            print(image_name)

if __name__=="__main__":
    main()