
from flask import Flask,render_template, redirect, url_for, request
app = Flask(__name__)

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/classify',methods = ['POST', 'GET'])
def classify():
   if request.method == 'POST':
      user = request.form['input']
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('input')
      return redirect(url_for('success',name = user))

@app.route('/classifyDoc')
def classifyDoc():
   return render_template('index.html')

if __name__ == '__main__':
   app.run(debug = True)
