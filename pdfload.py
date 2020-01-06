# importing required modules 
import PyPDF2 
import os
from shutil import copyfile

# creating a pdf file object 



file_dir = "./paper_list/"
out_pdf = "./pdf/"
out_html = "./html/"

for files in os.listdir(file_dir):
	flag = 0
	filename = files
	print(filename)
	if filename != ".DS_Store":
		word = str(filename).replace(".pdf",'')
		filename = file_dir + str(filename)

		pdfFileObj = open(filename, 'rb') 

		try:
			# creating a pdf reader object 
			pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
			  
			# printing number of pages in pdf file 
			print(pdfReader.numPages) 
			  
			# creating a page object 
			pageObj = pdfReader.getPage(0) 
			  
			# extracting text from page 
			#print(pageObj.extractText())  
			# closing the pdf file object 
			print(files,"is a pdf file")
			outname = out_pdf + files
		except:
			print(files,"is not a pdf file")
			outname = out_html + word + ".html"
		copyfile(filename,outname)


		pdfFileObj.close() 


