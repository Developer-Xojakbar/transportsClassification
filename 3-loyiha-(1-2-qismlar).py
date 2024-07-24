import torch
import fastai
from fastai.vision.all import *
from ipywidgets import widgets

# JUPYTERDA ISHLAYDI
# file da qanday komandalar borligini aniqlaydi 
# !cd OIDv4_ToolKit && python main.py -h
#%%
# %cd OIDv4_ToolKit
# !pip install -r requirements.txt
# !pip install urllib3==1.25.10

# !rm -rf OID
# !python3 main.py -y downloader --classes Car Airplane Boat --type_csv train --limit 10
# !cd OIDv4_ToolKit && python main.py -h OIDv4_ToolKit && python3 main.py downloader --Dataset /content --classes Car Airplane Boat --type_csv train --limit 200
# !cd OIDv4_ToolKit && python main.py -h OIDv4_ToolKit && python3 main.py downloader --Dataset "/Users/xojakbar/Desktop/Rabota/MohirDEV LEARNING/Data Science va sun'iy intellekt/20. Deep Learning" --classes Car Airplane Boat --type_csv train --limit 200
#%%


# path 
path = Path('images')
# path.ls() # folder tekshiradi
# fls = get_image_files(path) # imagelar sonini tekshiradi
# failed = verify_images(path) # failed ocurrapted imagelarni kursatadi

# Datablock yaratamiz
transports = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(224)
)

# Dataloader yaratamiz
dls = transports.dataloaders(path)

# datasetni tekshirish
# dls.train.show_batch(max_n=32, n_rows=4)

# O'qitish (TRAIN)
learn = cnn_learner(dls,resnet34,metrics=accuracy)
learn.fine_tune(4)


# GRAFIK tekshirish
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# eng katta xatoni kursatadi GRAFIKDA
interp.plot_top_losses(5,nrows=1)

# Modelni tekshirib kuramiz
upload = widgets.FileUpload()
upload

img = PILImage.create(upload.data[-1])
pred, pred_id, probs = learn.predict(img)
print(f'Bashorat:{pred}')
print(f'Ehtimollik:{probs[pred_id]*100:.1f}%')
img



# LOYIHA 2-QISM
# o'qitilgan modelni saqlab olish
learn.export('transport_model.pkl')

# o'qitilgan modelni tekshirib kurish
model = load_learner('transport_model.pkl')
model.predict(img)


# DEPLOY qilish VSCODE yozish kere
# python -m venv venv
# .\venv\Scripts\activate
# pip install fastai==2.5.3 streamlit
# davomi app.py degan file da



















# %%
