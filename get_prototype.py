import urllib.request
import shutil

url = "http://www.freebooks4doctors.com/link.php?id=1429"



# Download the file from `url`, save it in a temporary directory and get the
# path to it (e.g. '/tmp/tmpb48zma.txt') in the `file_name` variable:
#file_name, headers = urllib.request.urlretrieve(url)
#print(file_name)

# Download the file from `url` and save it locally under `file_name`:
file_name = "test.pdf"
with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)