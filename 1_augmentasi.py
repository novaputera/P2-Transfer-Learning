import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

datagen = ImageDataGenerator(
    rotation_range=90,       
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,          
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=True
)

kelas_benda = {
    'step_down': 'foto_asli/step_down_1.jpg', 
    'servo_driver': 'foto_asli/servo_driver_1.jpg'
}
jumlah_simulasi = 50

# Ambil lokasi folder saat ini secara otomatis
current_dir = os.getcwd()

for kelas, path_asli in kelas_benda.items():
    # Buat jalur folder yang pasti akurat
    folder_kelas = os.path.join(current_dir, 'dataset_raw', kelas)
    
    # PAKSA BUAT FOLDER: Jika foldernya belum ada, Python akan otomatis membuatnya di sini
    os.makedirs(folder_kelas, exist_ok=True)
    
    # Path foto asli
    path_asli_absolut = os.path.join(current_dir, path_asli)
    
    if os.path.exists(path_asli_absolut):
        print(f"Memproses {kelas}...")
        img = load_img(path_asli_absolut)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=folder_kelas, save_prefix=kelas, save_format='jpg'):
            i += 1
            if i >= jumlah_simulasi:
                break
        print(f"SUKSES: 50 gambar augmentasi untuk {kelas} sudah tersimpan!")
    else:
        print(f"ERROR: File {path_asli_absolut} tidak ditemukan! Cek kembali nama filenya.")