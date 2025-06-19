from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def canny_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    edges = cv2.Canny(blurred, 100, 200)
    return edges

def prewitt_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    img_prewittx = cv2.filter2D(gray, -1, kernelx)
    img_prewitty = cv2.filter2D(gray, -1, kernely)
    edges = cv2.add(img_prewittx, img_prewitty)
    return edges

def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(x, y)
    return cv2.convertScaleAbs(edges)

def roberts_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1, 0], [0, -1]])
    kernely = np.array([[0, 1], [-1, 0]])
    x = cv2.filter2D(gray, -1, kernelx)
    y = cv2.filter2D(gray, -1, kernely)
    edges = cv2.addWeighted(x, 0.5, y, 0.5, 0)
    return edges

@app.route('/', methods=['GET', 'POST'])
def main():
    
    for file in ['input.jpg', 'output.jpg', 'input_threshold.jpg', 'binary.jpg', 'binary_inv.jpg', 'trunc.jpg', 
                 'tozero.jpg', 'tozero_inv.jpg', 'otsu.jpg', 'original.jpg', 'histogram.png', 'input_morph.jpg', 'output_morph.jpg', 'input_morph2.jpg', 'output_morph2.jpg']:
        path = os.path.join(UPLOAD_FOLDER, file)
        if os.path.exists(path):
            os.remove(path)
    
    output_exists = False

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, 'input.jpg')
            file.save(filepath)

            img = cv2.imread(filepath)
            method = request.form.get("pilih")

            if method == "canny":
                edges = canny_edge_detection(img)
            elif method == "prewitt":
                edges = prewitt_edge_detection(img)
            elif method == "sobel":
                edges = sobel_edge_detection(img)
            elif method == "roberts":
                edges = roberts_edge_detection(img)
            else:
                edges = img

            output_path = os.path.join(UPLOAD_FOLDER, 'output.jpg')
            cv2.imwrite(output_path, edges)
            output_exists = True

    return render_template('index.html', output_exists=output_exists)

@app.route('/reset', methods=['POST'])
def reset():
    input_path = os.path.join(UPLOAD_FOLDER, 'input.jpg')
    output_path = os.path.join(UPLOAD_FOLDER, 'output.jpg')

    if os.path.exists(input_path):
        os.remove(input_path)
    if os.path.exists(output_path):
        os.remove(output_path)

    return redirect(url_for('main'))


@app.route('/threshold', methods=['GET', 'POST'])
def threshold():
    for file in ['input.jpg', 'output.jpg', 'input_threshold.jpg', 'binary.jpg', 'binary_inv.jpg', 'trunc.jpg', 
                 'tozero.jpg', 'tozero_inv.jpg', 'otsu.jpg', 'original.jpg', 'histogram.png', 'input_morph.jpg', 'output_morph.jpg', 'input_morph2.jpg', 'output_morph2.jpg']:
        path = os.path.join(UPLOAD_FOLDER, file)
        if os.path.exists(path):
            os.remove(path)

    output_exists = False
    
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'input_threshold.jpg')
            file.save(filepath)

            img = cv2.imread(filepath, 0)
            otsu_val, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, binary = cv2.threshold(img, otsu_val, 255, cv2.THRESH_BINARY)
            _, binary_inv = cv2.threshold(img, otsu_val, 255, cv2.THRESH_BINARY_INV)
            _, trunc = cv2.threshold(img, otsu_val, 255, cv2.THRESH_TRUNC)
            _, tozero = cv2.threshold(img, otsu_val, 255, cv2.THRESH_TOZERO)
            _, tozero_inv = cv2.threshold(img, otsu_val, 255, cv2.THRESH_TOZERO_INV)

            size = (600, 600)
            resized = cv2.resize(img, size)
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'original.jpg'), resized)
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'binary.jpg'), cv2.resize(binary, size))
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'binary_inv.jpg'), cv2.resize(binary_inv, size))
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'trunc.jpg'), cv2.resize(trunc, size))
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'tozero.jpg'), cv2.resize(tozero, size))
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'tozero_inv.jpg'), cv2.resize(tozero_inv, size))
            cv2.imwrite(os.path.join(UPLOAD_FOLDER, 'otsu.jpg'), cv2.resize(thresh_otsu, size))

            # Histogram
            from matplotlib import pyplot as plt
            plt.figure(figsize=(6, 3))
            plt.hist(resized.ravel(), bins=256, range=(0, 256), color='gray')
            plt.axvline(x=otsu_val, color='r', linestyle='--', label=f'Otsu = {int(otsu_val)}')
            plt.title("Histogram")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(UPLOAD_FOLDER, 'histogram.png'))
            plt.close()

            output_exists = True

    return render_template('threshold.html', output_exists=output_exists)

@app.route('/reset_threshold', methods=['POST'])
def reset_threshold():
    input = os.path.join(UPLOAD_FOLDER, 'input_threshold.jpg')
    binary = os.path.join(UPLOAD_FOLDER, 'binary.jpg')
    binary_inv = os.path.join(UPLOAD_FOLDER, 'binary_inv.jpg')
    trunc = os.path.join(UPLOAD_FOLDER, 'trunc.jpg')
    tozero = os.path.join(UPLOAD_FOLDER, 'tozero.jpg')
    tozero_inv = os.path.join(UPLOAD_FOLDER, 'tozero_inv.jpg')
    otsu = os.path.join(UPLOAD_FOLDER, 'otsu.jpg')
    original = os.path.join(UPLOAD_FOLDER, 'original.jpg')
    hist = os.path.join(UPLOAD_FOLDER, 'histogram.png')

    for path in [input, hist, binary, binary_inv, trunc, tozero, tozero_inv, otsu, original]:
        if os.path.exists(path):
            os.remove(path)

    return redirect(url_for('threshold'))


@app.route('/morphology', methods=['GET', 'POST'])
def morphology():
    
    for file in ['input.jpg', 'output.jpg', 'input_threshold.jpg', 'binary.jpg', 'binary_inv.jpg', 'trunc.jpg', 
                 'tozero.jpg', 'tozero_inv.jpg', 'otsu.jpg', 'original.jpg', 'histogram.png', 'input_morph.jpg', 'output_morph.jpg', 'input_morph2.jpg', 'output_morph2.jpg']:
        path = os.path.join(UPLOAD_FOLDER, file)
        if os.path.exists(path):
            os.remove(path)

    output_exists = False
    if request.method == 'POST':
        f = request.files['image']
        operation = request.form['operation']
        shape = request.form['shape']

        filepath = os.path.join(UPLOAD_FOLDER, 'input_morph.jpg')
        f.save(filepath)

        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, int(0.8 * 255), 255, cv2.THRESH_BINARY)

        if shape == 'disk':
            size = 5
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size*2+1, size*2+1))
        elif shape == 'diamond':
            size = 3
            se = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=np.uint8)
        elif shape == 'line':
            length = 10
            angle = 45
            line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
            M = cv2.getRotationMatrix2D((length // 2, 0), angle, 1)
            rotated = cv2.warpAffine(line_kernel, M, (length, length))
            _, se = cv2.threshold(rotated, 127, 1, cv2.THRESH_BINARY)
            se = se.astype(np.uint8)
        elif shape == 'rectangle':
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        elif shape == 'square':
            size = 4
            se = np.ones((size, size), dtype=np.uint8)
        elif shape == 'octagon':
            se = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
            ], dtype=np.uint8)
        elif shape == 'sphere':
           se = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
            ], dtype=np.uint8)
        else:
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        if operation == 'erode':
            result = cv2.erode(binary, se, iterations=1)
        else:
            result = cv2.dilate(binary, se, iterations=1)

        output_path = os.path.join(UPLOAD_FOLDER, 'output_morph.jpg')
        cv2.imwrite(output_path, result)
        output_exists = True

    return render_template('morphology.html', output_exists=output_exists)

@app.route('/reset_morphology', methods=['POST'])
def reset_morphology():
    input_path = os.path.join(UPLOAD_FOLDER, 'input_morph.jpg')
    output_path = os.path.join(UPLOAD_FOLDER, 'output_morph.jpg')

    if os.path.exists(input_path):
        os.remove(input_path)
    if os.path.exists(output_path):
        os.remove(output_path)

    return redirect(url_for('morphology'))

@app.route("/morphology2", methods=["GET", "POST"])
def morphology2():

    for file in ['input.jpg', 'output.jpg', 'input_threshold.jpg', 'binary.jpg', 'binary_inv.jpg', 'trunc.jpg', 
                 'tozero.jpg', 'tozero_inv.jpg', 'otsu.jpg', 'original.jpg', 'histogram.png', 'input_morph.jpg', 'output_morph.jpg', 'input_morph2.jpg', 'output_morph2.jpg']:
        path = os.path.join(UPLOAD_FOLDER, file)
        if os.path.exists(path):
            os.remove(path)

    output_exists = False

    if request.method == 'POST':
        f = request.files['image']
        method = request.form['method']

        filepath = os.path.join(UPLOAD_FOLDER, 'input_morph2.jpg')
        f.save(filepath)

        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if method == 'skeleton':
            skeleton = np.zeros(binary.shape, np.uint8)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            temp = np.copy(binary)
            while True:
                eroded = cv2.erode(temp, element)
                opened = cv2.dilate(eroded, element)
                temp2 = cv2.subtract(temp, opened)
                skeleton = cv2.bitwise_or(skeleton, temp2)
                temp = eroded.copy()
                if cv2.countNonZero(temp) == 0:
                    break
            result = skeleton

        elif method == 'thinning':
            result = cv2.ximgproc.thinning(binary)

        elif method == 'boundary':
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            eroded = cv2.erode(binary, element)
            result = cv2.subtract(binary, eroded)

        elif method == 'regionfill':
            im_floodfill = binary.copy()
            h, w = binary.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(im_floodfill, mask, (0,0), 255)
            im_floodfill_inv = cv2.bitwise_not(im_floodfill)
            result = cv2.bitwise_or(binary, im_floodfill_inv)

        elif method == 'thickening':
            binary_inv = cv2.bitwise_not(binary)
            thin = cv2.ximgproc.thinning(binary_inv)
    
            prev = np.zeros_like(thin)
            curr = thin.copy()
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

            while True:
                dilated = cv2.dilate(curr, element)
                thickened = cv2.bitwise_and(binary_inv, dilated)
                if np.array_equal(thickened, prev):
                    break
                prev = curr
                curr = thickened

            result = cv2.bitwise_not(curr)

        elif method == 'convexhull':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result = np.zeros_like(binary)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 5000:
                    hull = cv2.convexHull(cnt)
                    cv2.drawContours(result, [hull], -1, 255, -1)

        elif method == 'pruning':
            from skimage.morphology import skeletonize
            from skimage.morphology import thin
            from skimage.morphology import remove_small_objects
            from skimage.util import invert
            from skimage import img_as_bool, img_as_ubyte

            binary_bool = img_as_bool(binary)
            skel = skeletonize(binary_bool)
            pruned = remove_small_objects(skel, min_size=20, connectivity=2)
            result = img_as_ubyte(pruned)

        else:
            result = binary

        output_path = os.path.join(UPLOAD_FOLDER, 'output_morph2.jpg')
        cv2.imwrite(output_path, result)
        output_exists = True

    return render_template('morphology2.html', output_exists=output_exists)

@app.route('/reset_morphology2', methods=['POST'])
def reset_morphology2():
    input_path = os.path.join(UPLOAD_FOLDER, 'input_morph2.jpg')
    output_path = os.path.join(UPLOAD_FOLDER, 'output_morph2.jpg')

    if os.path.exists(input_path):
        os.remove(input_path)
    if os.path.exists(output_path):
        os.remove(output_path)

    return redirect(url_for('morphology2'))

if __name__ == '__main__':
    app.run(debug=True)
