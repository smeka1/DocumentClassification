
import os.path
from flask import Flask,render_template, request, jsonify
import classify_csv as clfpy

app = Flask(__name__)


@app.route('/api')
def index():
   return """
   <html> 
   <h2>Welcome to Sriram's API!</h2>
   <body><h3>These are your available endpoints:</h3><br/>
   <a href='/classify'>Run requests to this URI</a> 
   <br/>   
   <a href='/classify_form'>Use input field here</a>
   </body>
   </html>
   """

@app.route('/api/classify',methods = ['POST', 'GET'])
def classify():
   if request.method == 'POST':
      content = request.form.get('words')
      if(request.headers.get('Content-Type') == 'text/html'):
         print(request.form)
         if(3==2):
            return "HTML response: "+ clfpy.predict([content])
      else:
         print(content)
         return "JSON response:\n"+ jsonify(prediction=clfpy.predict([content])) 
   else:
      content = request.args.get('words')
      if(request.headers.get('Content-Type') == 'text/html'):
         #print(content)
         return "HTML response:\n"+ clfpy.predict([user])
      else:
         return jsonify (prediction=clfpy.predict([user]))
   return "End of function"   

@app.route('/api/classify_form')
def classifyDoc():
   return render_template('index.html')

if __name__ == '__main__':
   app.run(debug = True)
