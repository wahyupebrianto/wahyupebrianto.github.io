---
layout: post
title: Membawa Model Deep learning ke Produksi dengan FastAI, Gradio dan Hugging Face
excerpt: "Panduan singkat untuk melakukan deployment model deep learning dengan fastAI, Gradio di Hugging Face. Penting untuk anda mengetahui bahwa penerapan deep learning bukan sekedar berlama-lama pada proses mengumpulkan data, melatih, dan mengevaluasi model. Akan tetapi kita juga perlu tau bagaimana proses iterasi End-to-End untuk membantu kita memahami gambaran proses yang bermuara ke produksi."
author: Wahyu Pebrianto
date: 2024-04-15 13:26:00
tags: formatting code
categories: DL
feature_image: https://www.freecodecamp.org/news/content/images/size/w2000/2024/01/Untitled-design.png
---

# Introduction
Tujuan utama blog ini yaitu memberi anda panduan singkat untuk melakukan deployment model deep learning dengan fastAI, Gradio di Hugging Face. Penting untuk anda mengetahui bahwa penerapan deep learning bukan sekedar berlama-lama pada proses mengumpulkan data, melatih, dan mengevaluasi model. Akan tetapi kita juga perlu tau bagaimana proses iterasi End-to-End untuk membantu kita memahami gambaran proses yang bermuara ke produksi.

Dalam panduan singkat ini, Anda akan menerapkan deep learning pada contoh tugas computer vision: image classification/recognition.
Anda akan memanfaatkan data yang telah dikumpulkan oleh para peneliti yang biasanya mereka sediakan secara online untuk tujuan pengembangan di bidang penelitian.
Dalam hal ini anda akan menggunakan data The Oxford-IIIT Pet sebanyak 7.349 kumpulan data gambar pada 37 kategori hewan peliharaan dengan sekitar 200 gambar untuk setiap kelas, yang berasal dari hasil riset universitas Oxford [[1]](https://www.robots.ox.ac.uk/~vgg/data/pets/).
Dalam segi Model, Anda akan memanfaatkan pre-trained model ResNet-50 [[2]](https://ieeexplore.ieee.org/document/7780459) yang telah dilatih dengan data skala besar ImageNet [[3]](https://ieeexplore.ieee.org/document/5206848) dan kita lakukan fine-tuning ke data Oxford-IIIT Pet. Untuk Interface, Anda akan menggunakan web sebagai user interface dengan memanfaatkan gradio [[4]](https://www.gradio.app/). Sementara untuk proses deployment, Anda akan memanfaatkan Hugging face [[5]](https://huggingface.co/).

**Catatan:** di blog ini saya tidak menjelaskan hal-hal fundamental penggunaan teknologi (Contoh: Git, Jupyter notebook, installasi package/kerangka kerja, dsb), juga teori dasar Deep Learning.

Berikut outline blog ini:
1. Apa itu fastAI?
2. Apa itu Hugging Face?
3. Proses Systematic:
   - Mulai dengan melatih model deep learning (Image Recognition) dengan FastAI,
   - Mari kita mempersiapkan gradio,
   - Mari kita melakukan deployment model kita ke produksi dengan memanfaatkan Hugging Space

# Apa itu FastAI?
FastAI [[6]](https://www.mdpi.com/2078-2489/11/2/108) adalah kerangka kerja deep learning yang bangun oleh Jeremy Howard dan Rachel Thomas untuk pendekatan praktis dalam penerapan deep learning. Kerangka kerja ini menawarkan hight-level API ke PyTorch dan menawarkan kursus pendampingan untuk membantu siswa dalam menerapkan deep learning dengan mudah. Untuk detail lengkapnya anda bisa membaca di sumber berikut [[6]](https://www.mdpi.com/2078-2489/11/2/108).

# Apa itu gradio?
Gradio [[4]](https://www.gradio.app/) adalah salah satu package python untuk membantu anda dengan cepat membuat aplikasi web untuk tujuan melakukan deploy model machine learning atau deep learning anda. Jangan kawatir !! selain bersifat open source, gradio benar-benar dapat membantu anda membuat aplikasi web dengan cepat karena fitur bawaan gradio meskipun anda tidak memiliki keterampilan dalam pemograman web.

# Apa itu Hugging Face?
Namanya membuat saya tertarik :D. Hungging Face [[5]](https://huggingface.co/)  atau secara umum dikenal sebagai HuggingSpace adalah platform tempat seorang developer juga penggemar AI  dapat membuat, menghosting, dan berbagi aplikasi machine learning/deep learning mereka dengan mudah.


# Proses Systematic: Mulai melatih model deep learning (Image Recognition) dengan FastAI
Dalam proses ini, Anda tidak perlu kawatir karena anda bisa dengan mudah menjalankannya menggunakan kerangka kerja FastAI sebagaimana jika anda mengikuti kode berikut:

```python
from fastai.vision.all import *
path = untar_data(URLs.PETS)
dls = ImageDataLoaders.from_name_re(path, get_image_files(path/'images'),
    pat='(.+)_\d+.jpg', item_tfms=Resize(460), 
    batch_tfms=aug_transforms(size=224, min_scale=0.75))
learn = vision_learner(dls, models.resnet50, metrics=accuracy)
learn.fine_tune(1)
learn.path = Path('.')
learn.export()
```

Penjelasan singkat kode di atas:

* `from fastai.vision.all import *`: ini memberi anda akses ke semua kelas yang anda perlukan untuk membuat model computer vision yang disediakan oleh FastAI.
* `path = untar_data(URLs.PETS)`: melakukan download kumpulan data dari koleksi data fast.ai (jika belum diunduh sebelumnya) ke server Anda, mengekstraknya (jika belum diekstraksi sebelumnya), dan mengembalikan Path objek dengan lokasi yang diekstraksi.
* `dls`: Merupakan dataLoader anda (didalamnya terdiri dari lokasi data gambar & label, fungsi transform yang berisi script yang diterapkan secara otomatis selama proses training).
* `learn`:
    * `vision_learner`: fungsi untuk memanggil kelas, yang mana vision_learner juga memiliki parameter pretrained, yang defaultnya adalah True. catatan: Saat menggunakan model yang telah dilatih sebelumnya (pretrained model), vision_learner akan menghapus lapisan terakhir, karena lapisan tersebut selalu disesuaikan secara khusus dengan tugas pelatihan asli (yaitu klasifikasi kumpulan data ImageNet), dan menggantinya dengan satu atau lebih lapisan baru dengan bobot yang random, dengan ukuran yang sesuai untuk kumpulan data Anda saat ini yang kita kenal dengan teknik transfer learning.
        > Catatan singkat: (Transfer Learning: Menggunakan model yang telah dilatih sebelumnya untuk tugas yang berbeda dengan model yang telah dilatih sebelumnya.).
        * `dls`: data apa yang ingin kita latih.
        * `resnet50`: kita memberi tahu kearngka kerja untuk membuat Convolutional Neural Network (CNN) dan menentukan arsitektur apa yang akan digunakan (yaitu jenis model apa yang akan dibuat, disini anda menggunakan arsitektur Resnet 50 lapisan).
    * `metrics=accuracy`: ini memberitahu metrics apa yang akan digunakan. 
        >Catatan singkat: Metrics adalah fungsi yang mengukur kualitas prediksi model menggunakan data set validasi, dan akan ditampilkan pada akhir setiap epoch.
    * `learn.fine-tuning(1)`: kita memberi tahu kerangka kerja cara menyesuaikan model, kita menggunakan fine_tune bukan fit karena kita memanfaatkan model yang telah dilatih sebelumnya. (1) artinya kita melatih model sebanyak satu epoch.
        > Catatan singkat: (Fine-tuning: Teknik Transfer Learning di mana parameter model yang telah dilatih sebelumnya diperbarui dengan pelatihan untuk periode tambahan menggunakan tugas yang berbeda dengan yang digunakan untuk pra-pelatihan).
* `learn.path = Path('.')` dan `learn.export()`: mengatur lokasi dan menyimpan model yang telah dilatih.

# Mari kita mulai mempersiapkan Gradio
**Langkah pertama**: Load model yang telah dilatih dan buat fungsi untuk prediksi.

```python
learn = load_learner('export.pkl')
labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}
```

**Langkah kedua**: Anda persiapkan Gradio untuk interface web dengan mengikuti kode dibawah dan jalankan di lokal komputer anda.

```python
import gradio as gr
image = gr.Image(type="pil")#gr.inputs.Image(shape=(192, 192))
label = gr.Label(num_top_classes=3)#gr.outputs.Label()
gr.Interface(fn=predict, inputs=image, outputs=label).launch(share=True)
```

Dalam hal ini Saya menggunakan Jupyter Lab, berikut tangkapan layar hasil setelah saya menjalankan di komputer lokal saya.

![Localgradio](/assets/images/blog-1/localgradio.jpg)

# Mari kita mulai melakukan deployment model ke Hugging Face

Langkah anda untuk Deploy model ke Hugging face:
* Langkah Pertama: anda perlu membuat akun di [[5]](https://huggingface.co/).
* Langkah kedua: Setelah akun dibuat, silahkan login kemudian anda dapat membuat new space:

![spacegradio](/assets/images/blog-1/newspace.png)

* Langkah Ketiga: Clone repository space yang dibuat (kita remote melalui directory lokal komputer) dengan menjalankan `git clone https://huggingface.co/spaces/WahyuLab/recognize` 
* Langkah Keempat: Setelah di Clone, silahkan tambahkan `4 file` di dalam direktori yang telah di clone: `app.py, requirements.txt, export.pkl, dan cat.jpg`:

Berikut untuk detail isi filenya:
File app.py:
```python
import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Pengenalan Ras Hewan Peliharaan Kucing dan Anjing"
description = "Selamat Datang di Aplikasi Pengklasifikasi ras Hewan Peliharaan yang di training pada kumpulan data Oxford Pets dengan Fastai :D"
article="<p style='text-align: center'><a href='https://wahyupebrianto.github.io/dl/2024/04/15/Gradio-HG-FastAI.html' target='_blank'>Blog post</a></p>"
examples = ['cat.jpg']

image = gr.Image(type="pil")#gr.inputs.Image(shape=(192, 192))
label = gr.Label(num_top_classes=3)#gr.outputs.Label()

gr.Interface(fn=predict,inputs=image,
             outputs=label,title=title,
             description=description,article=article,examples=examples).launch()
```

File requirement.txt:
```python
fastai
scikit-image
```
Kemudian silahkan tambahkan File export.pkl dan satu contoh gambar dengan nama cat.jpg

* Langkah Kelima: Push ke-empat file ke repository gradio:
![spacegradio](/assets/images/blog-1/file.png)
> Catatan: File model export.pkl terlalu besar untuk ditangani oleh git. Jadi anda perlu menggunakan git-lfs (harus Anda instal terlebih dahulu). Disini saya menggunakan WSL dalam proses installasinya dengan menjalankan apt-get install git-lfs. Setelah anda berhasil melakukan installasi silahkan push ke direktori hingga muncul seperti gambar di atas pada huggingspace dengan menjalankan command berikut:

```python
#Langkah pertama:
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git commit -m "update .gitattributes jadi git lfs akan melacak .pkl files"
#Langkah kedua push ke Space:
git commit -am "mari kita terapkan ke huggingface spaces"
git push
```
Setelah proses di atas Anda ikuti, Anda dapat menunggu huggingspaces melakukan build, Anda dapat memeriksanya di bagian logs (tunggu hingga status  menjadi running).

![spacegradio](/assets/images/blog-1/running.png)

Hasil Deployment model deep leaning: Anda dapat memerika hasilnya, Jika berhasil, dan berikut contoh tangkapan layar implementasinya:

![spacegradio](/assets/images/blog-1/deployment1.png)

![spacegradio](/assets/images/blog-1/deployment2.png)


Hasil dari blog ini anda dapat memeriksanya dengan mengikuti link [ini](https://huggingface.co/spaces/WahyuLab/recognize)