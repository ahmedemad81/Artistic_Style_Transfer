import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
from os import access,R_OK
from tkinter import Label,messagebox
import style_transfer 
import numpy as np
from skimage import io
import time


def start_button_fn(content_path, style_path , sigma_r, sigma_s , canny_sigma , canny_filter_size , closing_iterations , dilation_iterations , kmean_k  , segmentation_mode, color_transfer_mode , LMAX  , PATCH_SIZES , SAMPLING_GAPS , IALG , IRLS_it , IRLS_r):
    if(content_path.get() == "" or style_path.get() == ""):
        messagebox.showinfo("Error", "Please select both content and style images first.")
        return
    patch_size = np.array([int(x) for x in PATCH_SIZES.split(',')])
    sampling_gaps = np.array([int(x) for x in SAMPLING_GAPS.split(',')])
    stylized_img, time_taken  = style_transfer.main(content_path.get(), style_path.get(),float(sigma_r) , float(sigma_s) , float (canny_sigma) , int (canny_filter_size), int (closing_iterations), int (dilation_iterations)  , int(kmean_k)  , segmentation_mode  , color_transfer_mode , int(LMAX) , patch_size , sampling_gaps , int (IALG) , int (IRLS_it)  , float (IRLS_r) )
    # Save the stylized image
    io.imsave('./output/stylized_img.jpg', (stylized_img.astype(np.float32)*255.0).astype(np.uint8))
    # Display the stylized image on the GUI
    output_img = ImageTk.PhotoImage(Image.open('./output/stylized_img.jpg').resize((400, 400)))
    output_label = Label(app, image=output_img)
    output_label.image = output_img
    output_label.grid(row=4, column=6, columnspan=2, padx=5,pady=10)
    time_taken_label = ctk.CTkLabel(app, text="Time Taken: " + str(round (time_taken,2)) + " seconds", font=("Arial", 15))
    time_taken_label.grid(row=4, column=8, columnspan=2, padx=10,pady=20)

def browse_button_content_fn():

    filename = filedialog.askopenfilename()
    if filename:
        content_path.set(filename)
        display_image(filename, row=4, column=0)

def browse_button_style_fn():

    filename = filedialog.askopenfilename()
    if filename:
        style_path.set(filename)
        display_image(filename, row=4, column=3)

def display_image( image_path, row, column, columnspan=3, padx=10,pady=10):
    # Load the selected image using Pillow
    image = Image.open(image_path)
    image = image.resize((400, 400))  # Resize the image if needed

    # Convert the image to Tkinter PhotoImage format
    tk_image = ImageTk.PhotoImage(image)

    # Create a label to display the image
    image_label = tk.Label(app, image=tk_image)
    image_label.grid(row=row, column=column, columnspan=columnspan, padx = 5,pady=pady)

    # Keep a reference to the image to avoid garbage collection issues
    image_label.image = tk_image

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("1600x900")
app.title("Artistic Style Transfer")
app.resizable(False, False)
app.grid_propagate(False)
###############VARS############################
content_path = ctk.StringVar()
style_path = ctk.StringVar()
# Write all the input for start button here as text variables
sigma_r = ctk.DoubleVar(value=0.05)
sigma_s = ctk.DoubleVar(value=10)
canny_sigma = ctk.DoubleVar(value=0.5)
canny_filter_size = ctk.DoubleVar(value=3)
closing_iterations = ctk.DoubleVar(value=2)
dilation_iterations = ctk.DoubleVar(value=4)
kmean_k = ctk.DoubleVar(value=2)
segmentation_mode = ctk.StringVar(value="watershed")
color_transfer_mode = ctk.StringVar(value="histogram")
L_max = ctk.DoubleVar(value=3)
IALG = ctk.DoubleVar(value=3)
IRLS_it = ctk.DoubleVar(value=3)
IRLS_r = ctk.DoubleVar(value=0.8)
patch_sizes = ctk.StringVar(value = '33,21,13,9')
sampling_gaps = ctk.StringVar(value = '28,18,8,5')
padding_mode = ctk.StringVar(value = 'edge')
time_taken = 0

#############################################

label = ctk.CTkLabel(app, text="Load Content Image", font=("Arial", 15))
label.grid(row=0,column=0,padx=10, pady=20)
browse_button_content = ctk.CTkButton(app, text="Browse", font=("Arial", 15),command=browse_button_content_fn,fg_color="grey")
browse_button_content.grid(row=0,column=1,padx=10, pady=20)

label = ctk.CTkLabel(app, text="Load Style Image", font=("Arial", 15))
label.grid(row=0,column=3,padx=10, pady=20)
browse_button_style = ctk.CTkButton(app, text="Browse", font=("Arial", 15),command=browse_button_style_fn,fg_color="grey")
browse_button_style.grid(row=0,column=4,padx=10, pady=20)

sigma_r = ctk.CTkEntry(app, textvariable=sigma_r, font=("Arial", 15))
sigma_r.grid(row=1, column=1, padx=10, pady=20)
sigma_r_label = ctk.CTkLabel(app, text="Sigma_r", font=("Arial", 15))
sigma_r_label.grid(row=1, column=0, padx=10, pady=20)

sigma_s = ctk.CTkEntry(app, textvariable=sigma_s, font=("Arial", 15))
sigma_s.grid(row=1, column=3, padx=10, pady=20)
sigma_s_label = ctk.CTkLabel(app, text="Sigma_s", font=("Arial", 15))
sigma_s_label.grid(row=1, column=2, padx=10, pady=20)

canny_sigma = ctk.CTkEntry(app, textvariable=canny_sigma, font=("Arial", 15))
canny_sigma.grid(row=2, column=5, padx=10, pady=20)
canny_sigma_label = ctk.CTkLabel(app, text="Canny Sigma", font=("Arial", 15))
canny_sigma_label.grid(row=2, column=4, padx=10, pady=20)

canny_filter_size = ctk.CTkEntry(app, textvariable=canny_filter_size, font=("Arial", 15))
canny_filter_size.grid(row=2, column=7, padx=10, pady=20)
canny_filter_size_label = ctk.CTkLabel(app, text="Canny Filter Size", font=("Arial", 15))
canny_filter_size_label.grid(row=2, column=6, padx=10, pady=20)

closing_iterations = ctk.CTkEntry(app, textvariable=closing_iterations, font=("Arial", 15))
closing_iterations.grid(row=2, column=1, padx=10, pady=20)
closing_iterations_label = ctk.CTkLabel(app, text="Closing Iterations", font=("Arial", 15))
closing_iterations_label.grid(row=2, column=0, padx=10, pady=20)

dilation_iterations = ctk.CTkEntry(app, textvariable=dilation_iterations, font=("Arial", 15))
dilation_iterations.grid(row=2, column=3, padx=10, pady=20)
dilation_iterations_label = ctk.CTkLabel(app, text="Dilation Iterations", font=("Arial", 15))
dilation_iterations_label.grid(row=2, column=2, padx=10, pady=20)


kmean_k = ctk.CTkEntry(app, textvariable=kmean_k, font=("Arial", 15))
kmean_k.grid(row=2, column=9, padx=10, pady=20)
kmean_k_label = ctk.CTkLabel(app, text="K-means", font=("Arial", 15))
kmean_k_label.grid(row=2, column=8, padx=10, pady=20)

segmentation_mode = ctk.CTkComboBox(app,values=["watershed","canny","otsu","kmeans"],variable=segmentation_mode, font=("Arial", 15))
segmentation_mode.grid(row=1, column=5, padx=10, pady=20)
segmentation_mode_label = ctk.CTkLabel(app, text="Segmentation Mode", font=("Arial", 15))
segmentation_mode_label.grid(row=1, column=4, padx=10, pady=20)

color_transfer_mode = ctk.CTkComboBox(app,values=["histogram","lab","mean"],variable=color_transfer_mode, font=("Arial", 15))
color_transfer_mode.grid(row=1, column=7, padx=10, pady=20)
color_transfer_mode_label = ctk.CTkLabel(app, text="Color Transfer Mode", font=("Arial", 15))
color_transfer_mode_label.grid(row=1, column=6, padx=10, pady=20)

L_max = ctk.CTkEntry(app, textvariable=L_max, font=("Arial", 15))
L_max.grid(row=3, column=5, padx=10, pady=20)
L_max_label = ctk.CTkLabel(app, text="Gaussian Pyramid", font=("Arial", 15))
L_max_label.grid(row=3, column=4, padx=10, pady=20)


patch_sizes = ctk.CTkComboBox(app,values=["33,21,13,9","33,21,13"],variable=patch_sizes, font=("Arial", 15))
patch_sizes.grid(row=3, column=1, padx=10, pady=20)
patch_sizes_label = ctk.CTkLabel(app, text="Patch Sizes", font=("Arial", 15))
patch_sizes_label.grid(row=3, column=0, padx=10, pady=20)

sampling_gaps = ctk.CTkComboBox(app,values=["28,18,8,5","28,18,8" ],variable=sampling_gaps, font=("Arial", 15))
sampling_gaps.grid(row=3, column=3, padx=10, pady=20)
sampling_gaps_label = ctk.CTkLabel(app, text="Sampling Gaps", font=("Arial", 15))
sampling_gaps_label.grid(row=3, column=2, padx=10, pady=20)

############# Start button #############
start_button = ctk.CTkButton(app, text="Start", font=("Arial", 15),fg_color="grey",command=lambda: start_button_fn(content_path, style_path , sigma_r.get() , sigma_s.get() , canny_sigma.get() , canny_filter_size.get() , closing_iterations.get() , dilation_iterations.get() , kmean_k.get()  , segmentation_mode.get() , color_transfer_mode.get() ,L_max.get() , patch_sizes.get() , sampling_gaps.get() , IALG.get() , IRLS_it.get() , IRLS_r.get() ))
start_button.grid(row=0, column=5, columnspan=2, pady=20)
############# Running loop #############
app.mainloop()
