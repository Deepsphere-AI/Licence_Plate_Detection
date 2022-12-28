import streamlit as st
from PIL import Image
import cv2
import imutils
import pytesseract
import easyocr
import numpy as np
import re
import pandas as pd

def main():
	st.set_page_config(layout="wide")
	global select2
	global select3
	global select4
	global file_uploaded
	output_file = open("output.txt",'w')
	global slect5
	global all_imgs
	global dict_file
	global features
	global Tesseract_HP
	global Easy_HP
	global Display_PSM 

	Display_PSM = None
	Easy_HP = None
	Tesseract_HP = None
	features = []
	all_imgs = []
	file_uploaded = []
	select2 = None 
	select3 = None
	select4 = None
	select5 = None
	dict_file = {}

	with open('style.css') as f:
		st.markdown (f"<style>{f.read()}</style>",unsafe_allow_html=True)

	def Reset_fun():
		pass
		st.session_state['key1']="Select the problem Statement"
		st.session_state['key2']="Library Used"
		st.session_state['key3']="Model Implemented"
		



	col1, col2, col3,col4,col5 = st.columns((2,2,7,2,2))
	with col1:
		st.write("")
	with col2:
		st.write("")
	with col3:
		img = Image.open("Deepsphere_image.png")
		st.image(img,use_column_width=True)
	with col4:
		st.write("")
	with col5:
		st.write("")

	st.markdown("<h1 style='text-align: center; color: Black;font-size: 29px;font-family:IBM Plex Sans;'>Learn to Build Industry Standard Data Science Applications</h1>", unsafe_allow_html=True)
	st.markdown("<p style='text-align: center; color: Blue;font-size: 29px;font-family:IBM Plex Sans;'>MLOPS Built On Google Cloud and Streamlit</p>", unsafe_allow_html=True)
	st.markdown("<p style='text-align: center; color: black; font-size:22px;font-family:IBM Plex Sans;'><span style='font-weight: bold'>Problem Statement: </span>Develop a Machine Learning Application for vehicle number plate Classfication</p>", unsafe_allow_html=True)
	st.markdown("______________________________________________________________________________________________________________________________________________")
	
	c11,c12,c13,c14,c15 = st.columns([0.25,1.5,2.75,0.25,1.75])
	with c12:
		# st.write("")
		st.write("")
		st.write("")
		st.markdown("#### **Problem Statement**")
	with c13:
		select1 = st.selectbox("",['Select the problem Statement','Number Plate Classification'],key = "key1")
	with c11:
		st.write("")
	with c14:
		st.write("")
	with c15:
		st.write("")

	st_list1 = ['Number Plate Classification']
	
	# c11,c12,c13,c14,c15 = st.columns([0.25,1.5,2.75,0.25,1.75])
	with c12:
		if select1 in st_list1:
			#st.write("")
			st.write("")
			st.write("")
			st.markdown("#### **Problem type**")
	with c13:
		if select1 in st_list1:
			select2 = st.selectbox("",['Select the problem type','Classification',])
	with c11:
		st.write("")
	with c14:
		st.write("")
	with c15:
		st.write("")


	st_list2 = ['Classification']
	# c11,c12,c13,c14,c15= st.columns([0.25,1.5,2.75,0.25,1.75])
	with c12:
		if select2 in st_list2:
			st.write("")
			st.write("")
			st.write("")
			st.markdown("#### **Model Selection**")
	with c13:
		if select2 in st_list2:
			select3 = st.selectbox("",['Select the Model','Tesseract-Ocr','Easy-Ocr'])
	with c11:
		st.write("")
	with c14:
		st.write("")
	with c15:
		st.write("")

	st_list3 = ['Tesseract-Ocr','Easy-Ocr']
	#c21,c22,c23,c24,c25 = st.columns([0.25,1.5,2.75,0.25,1.75])
	with c12:
		if select3 in st_list3:
			st.write("")
			st.write("")
			st.write("")
			st.markdown("#### **Upload File**")
	with c13:
		if select3 in st_list3:
			file_uploaded = st.file_uploader("Choose a image file", type=["JPG",'JFIF','JPEG','PNG','TIFF',],accept_multiple_files=True)
			if file_uploaded is not None:
				for file in file_uploaded:
				    # Convert the file to an opencv image.
				    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
				    all_imgs.append(cv2.imdecode(file_bytes, 1))
	with c11:
		st.write("")
	with c14:
		st.write("")
	with c15:
		if select3 in st_list3:
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			st.write("")
			select4 = st.button('Preview')

	if select4 is True:
		cd1,cd2,cd3,cd4,cd5 = st.columns((2,2,2,2,2))
		if len(all_imgs) >5:
			Display_Images= all_imgs[0:5]
			for i in range(len(Display_Images)):
				with cd1:
					st.image(all_imgs[i])
				with cd2:
					st.image(all_imgs[i+1])
				with cd3:
					st.image(all_imgs[i+2])
				with cd4:
					st.image(all_imgs[i+3])
				with cd5:
					st.image(all_imgs[i+4])
					break
		else:
			# CE1,CE2,CE3 = st.columns((6,7,2))
			with c11:
				st.write("")
			with c12:
				st.write("")
			with c13:
				st.write("#### Upload atleast 5 Images")
			with c14:
				st.write("")
			with c15:
				st.write("")
				
	c21,c22,c23,c24,c25 = st.columns([0.25,1.5,2.75,0.25,1.75])
	with c22:
		if len(file_uploaded)>=1:
			st.write("")
			st.write("")
			st.write("")
			#st.write("")
			st.markdown("#### **Feature Engineering**")
	with c23:
		if len(file_uploaded)>=1:
			st.write("")
			features = st.multiselect("Image Features",["Licence Number","State"])
	with c21:
		st.write("")
	with c24:
		st.write("")
	with c25:
		st.write("")

	# c51,c52,c53,c54,c55 = st.columns([0.25,1.5,2.75,0.25,1.75])
	with c22:
		if len(file_uploaded)>=1:
			st.write("")
			st.write("")
			#st.write("")
			st.markdown("#### **Hyper Parameter Tunning**")
	with c23:
		if len(file_uploaded)>=1 and select3 == 'Tesseract-Ocr':
			Tesseract_HP = st.selectbox("Page segmentation modes(PSM)",["Select the value:Best is 6", 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
		elif len(file_uploaded)>=1 and select3 == 'Easy-Ocr':
			Easy_HP = st.selectbox("HperParameters: Select Confidence_Threshold",["How Confidence should be the model with predicted text :: 0.1 is 10 percent",0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

	with c21:
		st.write("")

	with c25:
		st.write("")
		st.write("")
		st.write("")
		st.write("")
		st.write("")
		st.write("")
		st.write("")
		st.write("")
		#st.write("")
		if len(file_uploaded)>=1 and select3 == 'Tesseract-Ocr':
			Display_PSM = st.button("PSM")
	with c24:
		st.write("")

	CPS1,CPS2,CPS3 = st.columns((4,8,2))
	with CPS1:
		st.write("")
	with CPS2:
		st.write("")
		if Display_PSM == True:
			page_segementation_codes = {"Value": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
											"Segementation_Method":["Orientation and script detection (OSD) only.",
											"Automatic page segmentation with OSD.",
											"Automatic page segmentation, but no OSD, or OCR. (not implemented)",
											"Fully automatic page segmentation, but no OSD. (Default)",
											"Assume a single column of text of variable sizes.",
											"Assume a single uniform block of vertically aligned text.",
											"Assume a single uniform block of text.",
											"Treat the image as a single text line.",
											"Treat the image as a single word.",
											"Treat the image as a single word in a circle.",
											"Treat the image as a single character.",
											"Sparse text. Find as much text as possible in no particular order.",
											"Sparse text with OSD.",
											"Raw line. Treat the image as a single text line"]}
			DF = pd.DataFrame(page_segementation_codes)
			DF.set_index("Value",inplace=True)
			st.dataframe(DF,width=700, height=500)

	c31,c32,c33,c34,c35 = st.columns([0.25,1.5,2.75,0.25,1.75])
	with c32:
		if len(file_uploaded)>=1:
			st.write("")
			st.write("")
			st.write("")
			st.markdown("#### **Model Engineering**")
	with c33:
		if len(file_uploaded)>=1:
			st.write("")
			st.write("")
			st.write("")
			select5 = st.button("Execute the Model")
	with c31:
		st.write("")
	with c34:
		st.write("")
	with c35:
		st.write("")
	if select5 is True:
		state_dictionary = {'AN': 'Andaman and Nicobar Islands', 
		                    'AP': 'Andhra Pradesh', 
		                    'AR': 'Arunachal Pradesh', 
		                    'AS': 'Assam',
		                    'BR': 'Bihar',
		                    'CH': 'Chandigarh', 
		                    'CT': 'Chhattisgarh', 
		                    'DN': 'Dadra and Nagar Haveli',
		                    'DD': 'Daman and Diu', 
		                    'DL': 'Delhi', 
		                    'GA': 'Goa',
		                    'GJ': 'Gujarat', 
		                    'HR': 'Haryana',
		                    'HP': 'Himachal Pradesh',
		                    'JK': 'Jammu and Kashmir', 
		                    'JH': 'Jharkhand', 
		                    'KA': 'Karnataka',
		                    'KL': 'Kerala', 
		                    'LD': 'Lakshadweep',
		                    'MP': 'Madhya Pradesh ', 
		                    'MH': 'Maharashtra',
		                    'MN': 'Manipur', 
		                    'ML': 'Meghalaya', 
		                    'MZ': 'Mizoram', 
		                    'NL': 'Nagaland', 
		                    'OR': 'Odisha', 
		                    'PY': 'Puducherry', 
		                    'PB': 'Punjab', 
		                    'RJ': 'Rajasthan',
		                    'SK': 'Sikkim', 
		                    'TN': 'Tamil Nadu', 
		                    'TG': 'Telangana', 
		                    'TR': 'Tripura', 
		                    'UP': 'Uttar Pradesh', 
		                    'UT': 'Uttarakhand', 
		                    'WB': 'West Bengal'}
		if select3=='Tesseract-Ocr' and type(Tesseract_HP) == int and len(features)==2:
			value = 1
			output_file = open("output.txt",'w')
			for input_image in all_imgs:
			    # Resizing the image
			    Resized_image = imutils.resize(input_image, width=300 )
			    #Converting the image to Gray Scale
			    gray_image = cv2.cvtColor(Resized_image, cv2.COLOR_BGR2GRAY)
			    # Filtering the image with bilateral filter
			    filtred_image = cv2.bilateralFilter(gray_image,15,15,15)
			    # Applying the canny edge detection method
			    canny_edge_image = cv2.Canny(filtred_image, 30, 200)
			    # Finding Contours
			    cnts,new = cv2.findContours(canny_edge_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
			    # Drewing Contours
			    image1= input_image.copy()
			    cv2.drawContours(image1,cnts,-1,(0,255,0),1,lineType=cv2.LINE_AA)
			    # Sorting the contours to size of 30
			    cnts = sorted(cnts, key = cv2.contourArea, reverse = True) [:30]
			    screenCnt = None
			    image2 = input_image.copy()
			    cv2.drawContours(image2,cnts,-1,(0,255,0),3)
			    #Finding the contours with four sides
			    i=7
			    flag = False
			    for c in cnts:
			        perimeter = cv2.arcLength(c, True)
			        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
			        if len(approx) == 4:
			          flag = True 
			          screenCnt = approx
			          x,y,w,h = cv2.boundingRect(c) 
			          new_img=Resized_image[y:y+h,x:x+w]
			          cv2.imwrite('./image.png',new_img)
			          i+=1
			          break
			        else:
			          cv2.imwrite('./image.png',filtred_image)
			    # Drawing the detected square contour on the plate
			    try:
			      cv2.drawContours(Resized_image, [screenCnt], -1, (0, 255, 0), 3)
			    except Exception as e:
			      pass
			    # Passing the image to tesseract to get the text
			    if flag == True:

			      Cropped_loc = './image.png'
			      plate = pytesseract.image_to_string(Cropped_loc, lang='eng',config =f'--oem 3 --psm {Tesseract_HP}')
			    else:

			      Cropped_loc = './image.png'
			      plate = pytesseract.image_to_string(Cropped_loc, lang='eng',config = f'--oem 3 --psm {Tesseract_HP}')
			    # Remove unwated characters from the text
			    filteredText = re.sub('[^A-Z0-9.]+', ' ',plate)
			    if filteredText == " " or len(filteredText)<5:
			      plate = pytesseract.image_to_string(filtred_image, lang='eng',config = f'--oem 3 --psm {Tesseract_HP}')
			      plate_text = re.sub('[^A-Z0-9.]+', ' ',plate)
			      string = plate.replace(" ","")
			      string_list = string[:2]
			      try:
			        state = state_dictionary[string_list]
			      except Exception as e:
			       state = "UNKNOWN"
			       try:
			        if state == "UNKNOWN":
			          plate = pytesseract.image_to_string(input_image, lang='eng',config ='--oem 3 --psm 6')
			          plate_text = re.sub('[^A-Z0-9.]+', ' ',plate)
			          string = plate_text.replace(" ","")
			          string_list = string[:2]
			          state = state_dictionary[string_list]
			       except:
			        state = 'UNKNOWN'
			      output_file.write(f"\n{value} Number plate: {string}  state: {state}\n")
			    elif filteredText != " ":
			      string = filteredText.replace(" ","")
			      string_list = string[:2]
			      try:
			        state = state_dictionary[string_list]
			      except Exception as e:
			        state ="UNKNOWN"
			        try:
			          if state == "UNKNOWN":
			            plate = pytesseract.image_to_string(input_image, lang='eng',config ='--oem 3 --psm 6')
			            plate_text = re.sub('[^A-Z0-9.]+', ' ',plate)
			            string = plate_text.replace(" ","")
			            string_list = string[:2]
			            state = state_dictionary[string_list]
			        except:
			          state = 'UNKNOWN'
			      
			      output_file.write(f"\n{value} Number plate: {string}  state: {state}\n")
			    value += 1
			output_file.close()
		elif select3 == 'Easy-Ocr' and type(Easy_HP)==float and len(features)==2:
				output_file = open("output.txt",'w')
				reader = easyocr.Reader(['en'])
				value = 1
				confidence_threshold = Easy_HP
				india = ['IND','INDIA']
				for input_image in all_imgs:
					ocr_results = reader.readtext(input_image)
					for detection in ocr_results:
						if detection[2] > confidence_threshold and detection[1] not in india:
							detected = detection[1].upper()
							text_detection = re.sub('[^A-Z0-9.]+', ' ',detected)
							text = text_detection.replace(" ","")
							state_text = text[:2]
							if state_text == 'HH':
								state_text = 'MH'
							elif state_text == 'HB':
								state_text = 'WB'
							elif state_text == 'PV':
								state_text = 'PY'
							elif state_text == '6J':
								state_text = 'GJ'
							elif state_text == 'IN':
								state_text = 'TN'
							state_list = ['AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CT', 'DN', 'DD', 'DL', 'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'LD', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OR', 'PY', 'PB', 'RJ', 'SK', 'TN', 'TG', 'TR','TS', 'UP', 'UT', 'WB']        
							try:
								if state_text in  state_list:
									state = state_dictionary[state_text]
									output_file.write(f"\n{value} Number Plate:    {text}      state:  {state}\n")
									value += 1
									break
								else:
									state = "UNKNOWN"
									output_file.write(f"\n{value} Number Plate: {text}        state:  {state}\n")
									value += 1
									break
							except Exception as e:
								pass
				output_file.close()
		else:
			if select3 !=None:
				CE1,CE2,CE3 = st.columns((6,7,2))
				with CE1:
					st.write("")
				with CE2:
					st.write("")
					if select5 is True and len(features) !=2:
						st.error("Select the feature in Image Features")
					if select5 is True and  select3 == 'Tesseract-Ocr' and type(Tesseract_HP)!= int :
						st.error("Select HyperParameter value")
					if select5 is True and select3 == 'Easy-Ocr'and (type(Easy_HP)!=float):
						st.error("Select HyperParameter value")
				with CE3:
					st.write("")

	output_file = open('output.txt','r')
	c61,c62,c63,c64 = st.columns([4,3,3,5])
	with c61:
		st.write("")
	with c62:
		st.write("")
		st.write("")
		if len(output_file.readline()) != 0 and select5 == True:
			st.success("Model Executed Successfully")
	with c63:
		if len(file_uploaded)>=1:
			st.write("")
			st.write("")
			select6 = st.download_button("Download",output_file,file_name="OutPut.txt",mime='text')
	with c64:
		st.write("")
	output_file.close()

	
	st.sidebar.selectbox("",['Library Used','Streamlit','Pandas','Opencv','Tesseract-Ocr','Easy-Ocr'],key='key2')
	st.sidebar.selectbox("",['Model Implemented','Tesseract-Ocr','Easy-Ocr'],key='key3')
	
	c51,c52,c53 = st.sidebar.columns((1,1,1))
	with c51:
		pass
	with c52:
		st.sidebar.button("clear/Reset",on_click=Reset_fun)
	with c53:
		pass



if __name__ == '__main__':
	main()
