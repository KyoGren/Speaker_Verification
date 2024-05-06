import tkinter as tk

import utils
from parse_config import config

data_config = None
def select_mode(mode):
    global data_config
    # Hide the mode selection frame
    mode_frame.pack_forget()

    # Show the method selection frame
    if mode == "TDSV":
        config.task = "tdsv"
        data_config = config.data.TD_SV_data
        tdsv_frame.pack()
    elif mode == "TISV":
        config.task = "tisv"
        data_config = config.data.TI_SV_data
        tisv_frame.pack()

def go_back(mode):
    # Hide the current frame
    if mode == "TDSV":
        tdsv_frame.pack_forget()
        mode_frame.pack()
    elif mode == "TISV":
        tisv_frame.pack_forget()
        mode_frame.pack()

def main():
    global mode_frame, tdsv_frame, tisv_frame

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create the main window 
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    root = tk.Tk()
    root.title("Welcome to my channel")  
    root.geometry("800x300")

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
# Create a frame for mode selection
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    mode_frame = tk.Frame(root)  
    mode_frame.pack()

    mode_label = tk.Label(mode_frame, text="Select mode:", font=("Comic Sans MS", 20)) # Create a label for mode selection
    mode_label.pack(pady=20)

    tdsv_mode = tk.Button(mode_frame, text="TDSV", command=lambda: select_mode("TDSV"), font=("Comic Sans MS", 14), width=30, height=4) # Create a radio button for TDSV mode
    tdsv_mode.pack()
    
    tisv_mode = tk.Button(mode_frame, text="TISV", command=lambda: select_mode("TISV"), font=("Comic Sans MS", 14), width=30, height=4)  # Create a radio button for TISV mode
    tisv_mode.pack()



# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a frame for tdsv method 
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    tdsv_frame = tk.Frame(root)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # create a label for method selection
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    method_label = tk.Label(tdsv_frame, text="Text-Dependent Speaker Verification", font=("Comic Sans MS", 20)) # Create a label for method selection
    method_label.grid(row=0, column=0, columnspan=4, pady=20)
    
    
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create labels and buttons for enroll method
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    enroll_label = tk.Label(tdsv_frame, text="Enroll:") # Create a label for enrollment 
    enroll_label.grid(row=1, column=0)

    record_button = tk.Button(tdsv_frame, text="Record Audio", command=lambda: utils.record("TDSV", "enroll"), font=("Comic Sans MS", 14), width=15, height=2) # Create a button for recording audio
    record_button.grid(row=1, column=1)

    upload_button = tk.Button(tdsv_frame, text="Upload File", command=lambda: utils.upload_file("TDSV", "enroll"), font=("Comic Sans MS", 14), width=15, height=2) # Create a button for uploading a file
    upload_button.grid(row=1, column=2)

    show_audio_button = tk.Button(tdsv_frame, text="Show Audio", command=lambda: utils.show_audio(data_config.enroll_path), font=("Comic Sans MS", 14), width=15, height=2) # Create a button for showing the audio
    show_audio_button.grid(row=1, column=3)
    
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create labels and buttons for eval method
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    eval_label = tk.Label(tdsv_frame, text="Eval:") # Create a label for enrollment
    eval_label.grid(row=2, column=0)

    record_button = tk.Button(tdsv_frame, text="Record Audio", command=lambda: utils.record("TDSV","eval"), font=("Comic Sans MS", 14), width=15, height=2) # Create a button for recording audio
    record_button.grid(row=2, column=1)

    upload_button = tk.Button(tdsv_frame, text="Upload File", command=lambda: utils.upload_file("TDSV", "eval"), font=("Comic Sans MS", 14), width=15, height=2) # Create a button for uploading a file
    upload_button.grid(row=2, column=2)
    
    show_audio_button = tk.Button(tdsv_frame, text="Show Audio", command=lambda: utils.show_audio(data_config.eval_path), font=("Comic Sans MS", 14), width=15, height=2) # Create a button for showing the audio
    show_audio_button.grid(row=2, column=3)
   
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create a button for processing the audio
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    process_button = tk.Button(tdsv_frame, 
                               text="Process", 
                               command=lambda: utils.check_similarity(config.task, "./demonstration/data/tdsv/enrollment_audio.wav", "./demonstration/data/tdsv/evaluation_audio.wav", result_tdsv), 
                               font=("Comic Sans MS", 14), 
                               width=15, height=2) # Create a button for uploading a file
    process_button.grid(row=3, column=1)

    # result_tdsv = tk.Label(tdsv_frame, text="", font=("Comic Sans MS", 14)) # Create a label for showing the result
    # result_tdsv.grid(row=3, column=2)
    result_tdsv = tk.Entry(tdsv_frame, font=("Comic Sans MS", 14), width=35)
    result_tdsv.grid(row=3, column=2, columnspan=2)
    
    back_button = tk.Button(tdsv_frame, text="Back", command=lambda: go_back("TDSV"), font=("Comic Sans MS", 10), height=1) # Create a back button
    back_button.grid(row=5, column=3)  # Adjust the row and column as needed
    
    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a frame for tisv method
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    tisv_frame = tk.Frame(root)

    method_label = tk.Label(tisv_frame, text="Text-Independent Speaker Verification", font=("Comic Sans MS", 20)) # Create a label for method selection
    method_label.grid(row=0, column=0, columnspan=4, pady=20)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create labels and buttons for enroll method
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    enroll_label = tk.Label(tisv_frame, text="Enroll:") # Create a label for enrollment
    enroll_label.grid(row=1, column=0)

    record_button = tk.Button(tisv_frame, text="Record Audio", command=lambda: utils.record("TISV","enroll"), font=("Comic Sans MS", 14), width=15, height=2) # Create a button for recording audio
    record_button.grid(row=1, column=1)

    upload_button = tk.Button(tisv_frame, text="Upload File", command=lambda: utils.upload_file("TISV", "enroll"), font=("Comic Sans MS", 14), width=15, height=2) # Create a button for uploading a file
    upload_button.grid(row=1, column=2)

    show_audio_button = tk.Button(tisv_frame, text="Show Audio", command=lambda: utils.show_audio(data_config.enroll_path), font=("Comic Sans MS", 14), width=15, height=2) # Create a button for showing the audio
    show_audio_button.grid(row=1, column=3)
    
    eval_label = tk.Label(tisv_frame, text="Eval:") # Create a label for enrollment
    eval_label.grid(row=2, column=0)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create labels and buttons for eval method
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    record_button = tk.Button(tisv_frame, text="Record Audio", command=lambda: utils.record("TISV","eval"), font=("Comic Sans MS", 14), width=15, height=2) # Create a button for recording audio
    record_button.grid(row=2, column=1)

    upload_button = tk.Button(tisv_frame, text="Upload File", command=lambda: utils.upload_file("TISV", "eval"), font=("Comic Sans MS", 14), width=15, height=2) # Create a button for uploading a file
    upload_button.grid(row=2, column=2)
    
    show_audio_button = tk.Button(tisv_frame, text="Show Audio", command=lambda: utils.show_audio(data_config.eval_path), font=("Comic Sans MS", 14), width=15, height=2) # Create a button for showing the audio
    show_audio_button.grid(row=2, column=3)

    process_button = tk.Button(tisv_frame, 
                               text="Process", 
                               command=lambda: utils.check_similarity(config.task, "./demonstration/data/tisv/enrollment_audio.wav", "./demonstration/data/tisv/evaluation_audio.wav", result_tisv), 
                               font=("Comic Sans MS", 14), 
                               width=15, height=2) 
    
    process_button.grid(row=3, column=1)

    # result_tisv = tk.Label(tisv_frame, text="", font=("Comic Sans MS", 14)) 
    # result_tisv.grid(row=3, column=2)
    result_tisv = tk.Entry(tisv_frame, font=("Comic Sans MS", 14), width=35)
    result_tisv.grid(row=3, column=2, columnspan=2)
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Create the back button 
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    back_button = tk.Button(tisv_frame, text="Back", command=lambda: go_back("TISV"), font=("Comic Sans MS", 10), height=1) # Create a back button
    back_button.grid(row=6, column=3)  # Adjust the row and column as needed
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()

