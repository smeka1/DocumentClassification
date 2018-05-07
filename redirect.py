
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
      user = request.form['words']
      if(request.headers.get('Content-Type') == 'text/html'):
         return "HTML response:\n"+ clfpy.predict([user])
      else:
         return "JSON response: "+ clfpy.predict([user]) 
   else:
      user = request.args.get('words')
      if(request.headers.get('Content-Type') == 'text/html'):
         return "HTML response:\n"+ clfpy.predict([user])
      else:
         return jsonify (prediction=clfpy.predict([user]))
      

@app.route('/api/classify_form')
def classifyDoc():
   return render_template('index.html')

if __name__ == '__main__':
   app.run(debug = True)
