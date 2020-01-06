from bs4 import BeautifulSoup
import re
import os

pattern = [r"\<.*?\>","^http.*/$"]
#pattern = []
ref_file = './article_list/325711.html'

input_dir = "./article_list/"
output_dir = "./out_list/"

spe_list = ["Article last reviewed", "All references are available in the  References", "This page was printed from","Retrieved from"]

file_list = []
out_list = []

for files in os.listdir(input_dir):
	flag = 0
	filename = files
	#print(filename)
	if filename != ".DS_Store" and filename != ref_file:

		filename = input_dir + str(filename)

		file_list.append(filename)

		outname = filename.replace(input_dir,output_dir)
		outname = outname.replace(".html",".txt")


		out_list.append(outname)





#html_file = ['./article_list/325712.html','./article_list/325713.html','./article_list/325714.html']
#out_file = ['./test_out1.html','./test_out2.html','./test_out3.html']


file = open(ref_file,'r')
load_file = file.read()
#print(load_file)

ref_list = []

soup = BeautifulSoup(load_file)

#soup = soup.prettify()

info = soup.find_all("p")

#print(info)

for info_instance in info:
	info_purify = str(info_instance)
	ref_list.append(info_purify)


for i in range(len(file_list)):
	file = open(file_list[i],'r')
	load_file = file.read()
	#print(load_file)

	soup = BeautifulSoup(load_file)

	#soup = soup.prettify()

	info = soup.find_all("p")

	#print(info)

	out = open(out_list[i],'a')

	for info_instance in info:
		info_purify = str(info_instance)
		if not(info_purify in ref_list):
			for abd in pattern:
				#print(re.findall(abd,info_purify))
				info_purify = re.sub(abd," ",info_purify)
			for spe in spe_list:
				info_purify = info_purify.split(spe)[0]
			print("Generating...", out_list[i])
			out.write(info_purify)



