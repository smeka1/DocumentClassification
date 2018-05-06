from flask import Flask, render_template

app = Flask(__name__) #, static_url_path='')

@app.route('/')
def index():
    return "Works!" #app.send_static_file('index.html')

@app.route('/classifyDoc')
def classify():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)

