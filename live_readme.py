import os
import sys
import time
import aspose.words as aw
from http.server import BaseHTTPRequestHandler, HTTPServer

class Watchdog(object):
    running = True
    polling_rate = 1

    def __init__(self, file,func=None,*args,**kwargs):
        self.cached_stamp = 0
        self.filename = file
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def lookup(self):
        stamp = os.stat(self.filename).st_mtime
        if stamp != self.cached_stamp:
            self.cached_stamp = stamp
            print("File changed.")
            if self.func is not None:
                self.func(*self.args,**self.kwargs)
    
    def watchloop(self):
        while self.running:
            try:
                time.sleep(self.polling_rate)
                self.lookup()
            except KeyboardInterrupt:
                print("Terminating watchdog.")
                break
            except FileNotFoundError:
                pass
            except:
                print(f"Unhandled error: {sys.exc_info()} ")

class MD2PDFConverter(object):
    def __init__(self,file,output):
        self.file = file
        self.document = None
        self.tmppdf = str(os.path.join(os.getcwd(),output))
    
    def convert(self):
        self.document = aw.Document(self.file)
        self.document.save("output.html")


class MyServer(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/pdf')
        self.send_header('Content-Disposition', 'attachment; filename="file.pdf"')
        self.end_headers()

        # not sure about this part below
        if os.path.exists(os.path.join(os.getcwd(),"README.pdf")):
            self.wfile.write(open(os.path.join(os.getcwd(),"README.pdf"), 'rb'))
        else:
            self.wfile.write(open(os.path.join(os.getcwd(),"README.md"), 'rb'))

def convert_and_refresh(converter:MD2PDFConverter,server:MyServer):
    converter.convert()

if __name__ == "__main__":
    INPUT_FILE = os.path.join(os.getcwd(),"README.md")
    OUTPUT_FILE = "README.pdf"

    converter = MD2PDFConverter(
        file=INPUT_FILE,
        output=OUTPUT_FILE,
    )
    

    watchdog = Watchdog(
        file=INPUT_FILE,
        func=convert_and_refresh(converter,None),
        )

    watchdog.watchloop()
    
   