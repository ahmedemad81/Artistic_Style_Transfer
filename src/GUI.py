import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
from os import access,R_OK
from tkinter import Label,messagebox
import style_transfer 
import numpy as np
from skimage import io

LMAX = 3 
IMG_SIZE = 400
PATCH_SIZES = np.array([33 ,21 , 13 , 9])
SAMPLING_GAPS = np.array([28 , 18 , 8 , 5])
IALG = 3
IRLS_it = 3
IRLS_r = 0.8
PADDING_MODE = 'edge'
content_weight = 0.8

def start_button_fn(content_path, style_path):

    if(content_path.get() == "" or style_path.get() == ""):
        messagebox.showinfo("Error", "Please select both content and style images first.")
        return
    stylized_img, time_taken = style_transfer.main(content_path.get(), style_path.get(),sigma_r = 0.05 , sigma_s = 10 , canny_sigma = 0.5 , canny_filter_size = 3 , closing_iterations = 3 , dilation_iterations = 3 , kmean_k = 2 , segmentation_mode = 'watershed' , color_transfer_mode = 'histogram' , LMAX = LMAX , PATCH_SIZES = PATCH_SIZES , SAMPLING_GAPS = SAMPLING_GAPS , IALG = IALG , IRLS_it = IRLS_it , IRLS_r = IRLS_r)
    io.imsave('./output/stylized_img.jpg', stylized_img)
    # Display the stylized image on the GUI
    output_img = ImageTk.PhotoImage(Image.open('./output/stylized_img.jpg').resize((400, 400)))
    output_label = Label(app, image=output_img)
    output_label.image = output_img
    output_label.grid(row=1, column=6, columnspan=2, padx=10,pady=20)
    time_taken_label = ctk.CTkLabel(app, text="Time Taken: " + str(time_taken) + " seconds", font=("Arial", 15))
    time_taken_label.grid(row=2, column=6, columnspan=2, padx=10,pady=20)

def browse_button_content_fn():

    filename = filedialog.askopenfilename()
    if filename:
        content_path.set(filename)
        print(content_path.get())
        display_image(filename, row=1, column=0)

def browse_button_style_fn():

    filename = filedialog.askopenfilename()
    if filename:
        style_path.set(filename)
        print(style_path.get())
        display_image(filename, row=1, column=3,pady=20)

def display_image( image_path, row, column, columnspan=1, padx=10,pady=10):
    # Load the selected image using Pillow
    image = Image.open(image_path)
    image = image.resize((400, 400))  # Resize the image if needed

    # Convert the image to Tkinter PhotoImage format
    tk_image = ImageTk.PhotoImage(image)

    # Create a label to display the image
    image_label = tk.Label(app, image=tk_image)
    image_label.grid(row=row, column=column, columnspan=columnspan, padx=10,pady=pady)

    # Keep a reference to the image to avoid garbage collection issues
    image_label.image = tk_image

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("1400x500")
app.title("Artistic Style Transfer")
app.resizable(False, False)
###############VARS############################
content_path = ctk.StringVar()
style_path = ctk.StringVar()
stylized_img =np.zeros((IMG_SIZE, IMG_SIZE, 3))
time_taken = 0
 
##############################################

label = ctk.CTkLabel(app, text="Load Content Image", font=("Arial", 15))
label.grid(row=0,column=0,padx=20, pady=20)
browse_button_content = ctk.CTkButton(app, text="Browse", font=("Arial", 15),command=browse_button_content_fn,fg_color="grey")
browse_button_content.grid(row=0,column=1,padx=20, pady=20)

label = ctk.CTkLabel(app, text="Load Style Image", font=("Arial", 15))
label.grid(row=0,column=3,padx=20, pady=20)
browse_button_style = ctk.CTkButton(app, text="Browse", font=("Arial", 15),command=browse_button_style_fn,fg_color="grey")
browse_button_style.grid(row=0,column=4,padx=20, pady=20)

############# Start button #############
start_button = ctk.CTkButton(app, text="Start", font=("Arial", 15),fg_color="grey",command=lambda: start_button_fn(content_path, style_path))
start_button.grid(row=0, column=7, columnspan=2, pady=20)
############# Running loop #############
app.mainloop()

# import tkinter as tk
# from tkinter import filedialog
# from PIL import Image, ImageTk

# class ImageStylerApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Image Styler App")

#         # Initialize variables to store image paths
#         self.content_image_path = None
#         self.style_image_path = None

#         # Content Image Section
#         self.content_label = tk.Label(root, text="Content Image")
#         self.content_label.grid(row=0, column=0, padx=10, pady=10)

#         self.content_browse_button = tk.Button(root, text="Browse", command=self.browse_content_image)
#         self.content_browse_button.grid(row=0, column=1, padx=10, pady=10)

#         # Style Image Section
#         self.style_label = tk.Label(root, text="Style Image")
#         self.style_label.grid(row=1, column=0, padx=10, pady=10)

#         self.style_browse_button = tk.Button(root, text="Browse", command=self.browse_style_image)
#         self.style_browse_button.grid(row=1, column=1, padx=10, pady=10)

#         # Start Button
#         self.start_button = tk.Button(root, text="Start", command=self.show_first_image)
#         self.start_button.grid(row=2, column=0, columnspan=2, pady=20)

#     def browse_content_image(self):
#         # Open a file dialog to select a content image
#         self.content_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])

#         # Display the selected content image on the GUI
#         if self.content_image_path:
#             self.display_image(self.content_image_path, row=0, column=2)

#     def browse_style_image(self):
#         # Open a file dialog to select a style image
#         self.style_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])

#         # Display the selected style image on the GUI
#         if self.style_image_path:
#             self.display_image(self.style_image_path, row=1, column=2)

#     def show_first_image(self):
#         # Display the first picked image (content image) on the GUI
#         if self.content_image_path:
#             self.display_image(self.content_image_path, row=2, column=0, columnspan=2, pady=20)
#         else:
#             tk.messagebox.showinfo("Error", "Please select a content image first.")

#     def display_image(self, image_path, row, column, columnspan=1, pady=10):
#         # Load the selected image using Pillow
#         image = Image.open(image_path)
#         image = image.resize((200, 200), Image.ANTIALIAS)  # Resize the image if needed

#         # Convert the image to Tkinter PhotoImage format
#         tk_image = ImageTk.PhotoImage(image)

#         # Create a label to display the image
#         image_label = tk.Label(self.root, image=tk_image)
#         image_label.grid(row=row, column=column, columnspan=columnspan, pady=pady)

#         # Keep a reference to the image to avoid garbage collection issues
#         image_label.image = tk_image

# root = tk.Tk()
# app = ImageStylerApp(root)
# root.mainloop()