from tkinter import *
from tkinter import messagebox
# from tkinter.ttk import *
from PIL import ImageTk, Image
import numpy
import scipy.special
import matplotlib.pyplot
import imageio
import os
from pandas import DataFrame
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog as fd

filename = ''
outputs = ''
class neuralNetwork:

    # initialise the neural network( khởi tạo mạng)
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer (đánh số cho các node của thừng lớp)
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who( ma trận trọng lượng liên kết)
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # các trọng số bên trong các mảng là w_i_j, trong đó liên kết là từ i đến nút j.
        # w11 w21
        # w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate( tỉ lệ học )
        self.lr = learningrate

        # activation function is the sigmoid function( kích hoạt bằng hàm sigmoid)
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    #  save neural network weights
    #  def save(self):
    #      numpy.save('saved_wih.npy', self.wih)
    #      numpy.save('saved_who.npy', self.who)
    #     pass
    # huấn luyện mạng
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array ( chuyển đổi ds đầu vào thành mảng 2d)
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer( tính toán tín hiệu vào lớp ẩn)
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer( tính toán các tín hiệu nổi lên từ lớp ẩn)
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer( tính toán tín hiệu vào cảu lớp đầu ra cuối cùng
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer( tính toán các tín hiệu nổi lên từ lớp đầu ra cuối cùng)
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)( lỗi lớp đầu ra là ( mục tiêu trừ đi thực tế)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        # lỗi lớp ẩn, phân chia theo trọng số, tập hợp lại các nút ẩn.
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        # cập nhập các trọng số cho các liên kết giữa các lớp đầu ra và ẩn.
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                            numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        # cập nhập các trọng số cho các liên kết giữa các lớp đầu vào và ẩn.
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                            numpy.transpose(inputs))
        # self.save()
        pass

    # truy vấn mạng (query)
    def query(self, inputs_list):
        # convert inputs list to 2d array( chuyển đổi ds đầu vào thành mảng 2d)
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer( tính toán tín hiêu trong lớp ẩn )
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer ( tính toán giá trị hiện ra từ lớp ẩn)
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer ( tính toán tín hiệu trong lớp đầu ra cuối cùng)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer ( tính toán giá trị hiên ra từ lớp đầu ra cuối cùng)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def load(self):
            self.wih = numpy.load('saved_wih.npy')
            self.who = numpy.load('saved_who.npy')
# ---------------------------------------------------------------------------


# def toggled():
#     print("The check button works.")
def btnOpenfile_click():
    global filename
    filename = fd.askopenfilename()
    print(filename)

def btnHelp_click():
    messagebox.showinfo("Help","If you have any questions. Please keep for yourself. I'm not Google")

def btnRefresh_click():
    out_put.delete(0,END)
    out_put1.delete(0, END)

def btnOpen_click():
    os.startfile('my_own_images/')

def btnChart_click():
    data1 = {'Number': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
             'Recognition_rate(%)': [outputs[0,0]*100,outputs[1,0]*100,outputs[2,0]*100,outputs[3,0]*100,outputs[4,0]*100,
                                     outputs[5,0]*100,outputs[6,0]*100,outputs[7,0]*100,outputs[8,0]*100,outputs[9,0]*100]
             }
    df1 = DataFrame(data1, columns=['Number', 'Recognition_rate(%)'])
    print(df1)

    root = tk.Tk()
    figure1 = plt.Figure(figsize=(6, 5), dpi=100)
    ax1 = figure1.add_subplot(111)
    bar1 = FigureCanvasTkAgg(figure1, root)
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    df1 = df1[['Number', 'Recognition_rate(%)']].groupby('Number').sum()
    df1.plot(kind='bar', legend=True, ax=ax1,color='g')
    ax1.set_title('Number Vs. Recognition_rate(%)')
    root.mainloop()

# def btnLogin_click():
#     user = tbUser.get()
#     passw = tbPass.get()
#     if (user=="admin" and passw == "123"):
#         messagebox.showinfo("Thong Bao"," Login complete ")

def btnRun_click():
    #number of input, hidden and output nodes(số các nude của từng lớp )
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # learning rate ( tỷ lệ học )
    learning_rate = 0.1

    # create instance of neural network ( tạo trường hợp của mạng )
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    n.load()
    # --------------------------------------------------------------------------------
    # # load the mnist training data CSV file into a list ( tải tệp CSV dữ liệu đào tạo mnist vào danh sách)
    # training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
    # training_data_list = training_data_file.readlines()
    # training_data_file.close()
    # -----------------------------------------------------------------------------------
    # train the neural network ( huấn luyện)

    # epochs is the number of times the training data set is used for training
    # số lần tập dữ liệu huấn luyện được sử dụng cho đào tạo
    # epochs = 10
    #
    # for e in range(epochs):
    #     # go through all records in the training data set( thông qua tất cả dữ liệu đào tạo)
    #        for record in training_data_list:
    #         # split the record by the ',' commas ( chia bản ghi bằng dấu phẩy )
    #         all_values = record.split(',')
    #         # scale and shift the inputs ( mở rộng quy mô và thay đổi đầu vào )
    #         inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #         # create the target output values (all 0.01, except the desired label which is 0.99)
    #         # tạo các giá trị đầu ra đích( tất cả 0,01, ngoại trừ nhãn mong muốn là 0,99)
    #         targets = numpy.zeros(output_nodes) + 0.01
    #         # all_values[0] is the target label for this record ( all_values[0] là mục tiêu)
    #         targets[int(all_values[0])] = 0.99
    #         n.train(inputs, targets)
    #         pass
    # pass
    # ------------------------------------------------------------------------------------
    # test the neural network with our own images ( kiểm tra mạng với ảnh của riêng chúng tôi)

    # load image data from png files into an array( tải dữ liệu hình ảnh từ các tập tin png vào một mảng)
    # print("loading ... my_own_images/2828_my_own_image.png")
    img_array = imageio.imread(filename, as_gray=True)

    # reshape from 28x28 to list of 784 values, invert values ( định hình lại từ 28x28 thành danh sách 784 giá trị, đảo ngược giá trị)
    img_data = 255.0 - img_array.reshape(784)

    # then scale data to range from 0.01 to 1.0( chia tỉ lệ dữ liệu trọng phạm vi 0,01 đến 1,0)
    img_data = (img_data / 255.0 * 0.99) + 0.01
    # print("min = ", numpy.min(img_data))
    # print("max = ", numpy.max(img_data))

    # plot image (hình ảnh chính )
    matplotlib.pyplot.imshow(img_data.reshape(28, 28), cmap='Greys', interpolation='None')



    # query the network ( truy vấn mạng)
    global outputs
    outputs = n.query(img_data)
    print(outputs)
    n = round(max(outputs)[0],2)*100
    print("correct percent:"+str(n)+"%")
    out_put1.insert(0,str(n)+"%")
    # the index of the highest value corresponds to the label
    # chỉ số của giá trị cao nhất tương ứng với nhãn
    label = numpy.argmax(outputs)
    print("network says ", label)
    out_put.insert(0, label)

win2 = Tk()
win2.title("My Project")
win2.iconphoto(False, tk.PhotoImage(file='ladybug.png'))
frame = LabelFrame(win2, text ="Menu",font=("Consolas", 14,"bold"),fg="black",padx=40)
frame.pack( side = TOP )


photo = PhotoImage(file = r"run.png")
photo1 = PhotoImage(file = r"open2.png")
photo2 = PhotoImage(file = r"exit.png")
photo3 = PhotoImage(file = r"chart.png")
photo4 = PhotoImage(file = r"open1.png")
photo5 = PhotoImage(file = r"refresh.png")
photo6 = PhotoImage(file = r"help.png")

photoimage = photo.subsample(2, 2)
photoimage1 = photo1.subsample(2, 2)
photoimage2 = photo2.subsample(3, 3)
photoimage3 = photo3.subsample(2, 2)
photoimage4 = photo4.subsample(2, 2)
photoimage5 = photo5.subsample(2, 2)
photoimage6 = photo6.subsample(2, 2)


btnRun = Button(frame, text="Run", font=("Consolas",12), fg="black",command=btnRun_click, pady=2, image = photoimage,
                    compound = LEFT)
btnOpenpaint = Button(frame, text="Open Paint", font=("Consolas",12), fg="black",command=btnOpen_click,padx=5, pady=2, image = photoimage1,
                    compound = LEFT)
btnQuit = Button(frame, text="Exit", font=("Consolas",12), fg="black", command=win2.quit,padx=5, pady=2, image = photoimage2,
                    compound = LEFT)
btnChart = Button(frame, text="Chart", font=("Consolas",12), fg="black", command=btnChart_click,padx=5, pady=2, image = photoimage3,
                    compound = LEFT)
btnOpenfile = Button(frame, text="Open File", font=("Consolas",12), fg="black",command=btnOpenfile_click,padx=5, pady=2, image = photoimage4,
                    compound = LEFT)
btnRefresh = Button(frame, text="Refresh", font=("Consolas",12), fg="black",command=btnRefresh_click,padx=5, pady=2, image = photoimage5,
                    compound = LEFT)
btnHelp = Button(frame, text="Help", font=("Consolas",12), fg="black",command=btnHelp_click, pady=2, image = photoimage6,
                    compound = LEFT)


btnRun.grid(row=0, column=0, pady=5)
btnOpenpaint.grid(row=0, column=1)
btnOpenfile.grid(row=0, column=2)
btnQuit.grid(row=0, column=3)
btnChart.grid(row=0,column=4)
btnRefresh.grid(row=0,column=5)
btnHelp.grid(row=0, column=6)

frame4 = LabelFrame(win2)
frame4.pack()
img = ImageTk.PhotoImage(Image.open("network.png"))
panel = Label(frame4, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")

frame2 = LabelFrame(win2, text="Network says", font=("Consolas", 20, "bold"), fg="black")
frame2.pack(side = LEFT)
out_put = Entry(frame2, width=30, borderwidth=5, font=("Consolas", 14, "bold"))
out_put.pack()

frame3 = LabelFrame(win2, text="Current percent", font=("Consolas", 20, "bold"), fg="black")
frame3.pack(side = RIGHT)
out_put1 = Entry(frame3, width=30, borderwidth=5, font=("Consolas", 14, "bold"))
out_put1.pack()

win2.mainloop()



