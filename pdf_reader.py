"""
Скрипт принимает имя pdf-файла, который необходимо сохранить постранично.
Если файл не найден - вывод соответствующего сообщения
"""
# import the necessary package
import PyPDF2


# get a filename
pdf_doc = input("Введите имя исходного pdf-файла:\n")
# if the file has been found, read it
try:
    with open(pdf_doc, 'rb') as file:
        pdf = PyPDF2.PdfFileReader(file)
        pages = pdf.getNumPages()
        for i in range(pages):
            new_file = pdf.getPage(i)
            output = PyPDF2.PdfFileWriter()
            output.addPage(new_file)
            output_name = 'output_' + str(i+1) + '.pdf'
            with open(output_name, 'wb') as out:
                output.write(out)
except FileNotFoundError:
    print("The file not found")
