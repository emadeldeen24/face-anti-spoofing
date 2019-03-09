import numpy as np

class CSK:
    def __init__(self):
        self.eta = 0.075
        self.sigma = 0.2
        self.lmbda = 0.01

    def init(self,frame,x1,y1,width,height):
        # Save position and size of bbox
        self.x1 = x1
        self.y1 = y1
        self.width = width if width%2==0 else width-1
        self.height = height if height%2==0 else height-1

        # Crop & Window
        self.x = self.crop(frame,x1,y1,self.width,self.height)

        # Generate regression target
        self.y = self.target(self.width,self.height)
        self.prev = np.unravel_index(np.argmax(self.y, axis=None), self.y.shape) # Maximum position

        # Training
        self.alphaf = self.training(self.x,self.y,self.sigma,self.lmbda)

    def update(self,frame):
        # Crop at the previous position (doubled size)
        z = self.crop(frame,self.x1,self.y1,self.width,self.height)

        # Detection
        responses = self.detection(self.alphaf,self.x,z,0.2)
        curr = np.unravel_index(np.argmax(responses, axis=None), responses.shape)
        dy = curr[0]-self.prev[0]
        dx = curr[1]-self.prev[1]

         # New position (left top corner)
        self.x1 = self.x1 - dx
        self.y1 = self.y1 - dy

        # Training
        xtemp = self.eta*self.crop(frame,self.x1,self.y1,self.width,self.height) + (1-self.eta)*self.x
        self.x = self.crop(frame,self.x1,self.y1,self.width,self.height)

        self.alphaf = self.eta*self.training(self.x,self.y,0.2,0.01) + (1-self.eta)*self.alphaf # linearly interpolated
        self.x = xtemp

        return self.x1, self.y1


    def dgk(self, x1, x2, sigma):
        c = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(x1)*np.conj(np.fft.fft2(x2))))
        d = np.dot(np.conj(x1.flatten(1)),x1.flatten(1)) + np.dot(np.conj(x2.flatten(1)),x2.flatten(1)) - 2*c
        k = np.exp(-1/sigma**2*np.abs(d)/np.size(x1))
        return k

    def training(self, x, y, sigma, lmbda):
        k = self.dgk(x, x, sigma)

        # y2 = np.zeros_like(k)
        # for i in range(3):
        #     y2[:,:,i] = y
        alphaf = np.fft.fft2(y)/(np.fft.fft2(k)+lmbda)
        return alphaf

    def detection(self, alphaf, x, z, sigma):
        k = self.dgk(x, z, sigma)
        responses = np.real(np.fft.ifft2(alphaf*np.fft.fft2(k)))
        return responses

    def window(self,img):
        height = img.shape[0]
        width = img.shape[1]

        j = np.arange(0,width)
        i = np.arange(0,height)
        J, I = np.meshgrid(j,i)
        window = np.sin(np.pi*J/width)*np.sin(np.pi*I/height)
        windowed = window*((img/255)-0.5)

        return windowed

    def crop(self,img,x1,y1,width,height):
        pad_y = [0,0]
        pad_x = [0,0]

        if (y1-height/2) < 0:
            y_up = 0
            pad_y[0] = int(-(y1-height/2))
        else:
            y_up = int(y1-height/2)

        if (y1+3*height/2) > img.shape[0]:
            y_down = img.shape[0]
            pad_y[1] = int((y1+3*height/2) - img.shape[0])
        else:
            y_down = int(y1+3*height/2)

        if (x1-width/2) < 0:
            x_left = 0
            pad_x[0] = int(-(x1-width/2))
        else:
            x_left = int(x1-width/2)

        if (x1+3*width/2) > img.shape[1]:
            x_right = img.shape[1]
            pad_x[1] = int((x1+3*width/2) - img.shape[1])
        else:
            x_right = int(x1+3*width/2)

        cropped = img[y_up:y_down,x_left:x_right]
        padded = np.pad(cropped,(pad_y,pad_x),'edge')
        windowed = self.window(padded)
        return windowed

    def target(self,width,height):
        double_height = 2 * height
        double_width = 2 * width
        s = np.sqrt(double_height*double_width)/16

        j = np.arange(0,double_width)
        i = np.arange(0,double_height)
        J, I = np.meshgrid(j,i)
        y = np.exp(-((J-width)**2+(I-height)**2)/s**2)

        return y

