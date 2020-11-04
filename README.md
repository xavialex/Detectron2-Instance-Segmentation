# Detectron2 Instance Segmentation

Streamlit App that performs object detection and instance segmentation powered by Detectron2.

![app_preview](https://user-images.githubusercontent.com/17023476/98106894-086ba000-1e9a-11eb-8479-a3679e36d64a.png)


## Dependencies

Running the application can be done following the instructions above:

1. To create a Python Virtual Environment (virtualenv) to run the code, type:

    ```python3 -m venv my-env```

2. Activate the new environment:
    * Windows: ```my-env\Scripts\activate.bat```
    * macOS and Linux: ```source my-env/bin/activate``` 

3. Install all the dependencies from *requirements.txt*:

    ```pip install -r requirements.txt```

## Use

Within the virtual environment:

```streamlit run app.py```

A web application will open in the prompted URL. The user should upload an image file (*jpg*, *jpeg*, *png*) with the button available. Then, the image will be fed to a model which will output tehe original image with the detections drawn on it.

There's another app:

```streamlit run app_discriminative.py```

It behaves as the other one, but includes the following options:

* **Select which classes to detect:** Multiselect to choose which of the classes that the model's been trained on are going to be used in the inference. 

## Acknowledgments

* [Detectron 2](https://github.com/facebookresearch/detectron2)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details