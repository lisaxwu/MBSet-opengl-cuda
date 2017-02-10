/* 
 * File:   MBSet.cu
 * 
 * Created on June 24, 2012
 * 
 * Purpose:  This program displays Mandelbrot set using the GPU via CUDA and
 * OpenGL immediate mode.
 * 
 */

#include <iostream>
#include <stack>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "Complex.cu"

#include <GL/freeglut.h>
#ifdef __APPLE__
    #include <GLUT/glut.h>
    #include <OpenGL/glext.h>
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
#else
    #include <GL/glut.h>
    #include <GL/glext.h>
    #include <GL/gl.h>
    #include <GL/glu.h>
#endif

// Size of window in pixels, both width and height
#define WINDOW_DIM            512

using namespace std;

// Initial screen coordinates, both host and device.
Complex minC(-2.0, -1.2);
Complex maxC(1.0, 1.8);
Complex* dev_minC;
Complex* dev_maxC;
const int maxIt = 2000; // Msximum Iterations

int updateRate = 50;
//-------------host------------------
int pixels = WINDOW_DIM * WINDOW_DIM;
int window = WINDOW_DIM;
unsigned int w = WINDOW_DIM;
int* p = &window;

int num=0;
int* ifMBSet;
int* itera;
float* min_r = &(minC.r);
float* min_i = &(minC.i);
float* max_r = &(maxC.r);
float* max_i = &(maxC.i);

GLfloat* image=NULL;
GLfloat* image2=NULL;
//------------------------------------





// Define the RGB Class
class RGB
{
public:
    RGB()
        : r(0), g(0), b(0) {}
    RGB(double r0, double g0, double b0)
        : r(r0), g(g0), b(b0) {}
public:
    double r;
    double g;
    double b;
};

RGB* colors = 0; // Array of color values

void InitializeColors()
{
  colors = new RGB[maxIt + 1];
  for(int i = 0; i < 4; ++i)
  {
    colors[maxIt] = RGB(drand48(), drand48(), drand48());
  }
  for (int i = 0; i < maxIt; ++i)
    {
      if (i < 5)
        { // Try this.. just white for small it counts
          colors[i] = RGB(1, 1, 1);
        }
      else
        {
          colors[i] = RGB(drand48(), drand48(), drand48());
        }
    }
  colors[maxIt] = RGB(); // black
}


//-------------------------- my code ---------------------------

__global__ void isMBpoint(float* min_r,float* min_i,float* max_r,float* max_i,int* window,int* ifMB,int* ite)
{
    Complex c(0.0,0.0);
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int id = x + y * blockDim.x * gridDim.x;
    int N = *window;

    if(id<N*N)
    {
      int px= id / N;
      int py= id % N;
      float minr = *min_r;
      float mini = *min_i;
      float maxr = *max_r;
      float maxi = *max_i;
      c.r = (minr) + (maxr - minr)*1.0/N * px;
      c.i = (mini) + (maxi - mini)*1.0/N * py;

      Complex Z(c);
      int result = 1;
      int i;
      for(i=1;i<=2000;i++)
      {
          Z=Z*Z+c;
          if (Z.magnitude2()>4)   //mag^2 > 4
          {
              result=0;
              break;
          }
          
      }

      ite[id]=i;
      ifMB[id]=result;
    }
}

void AddRange(float minr,float mini,float maxr, float maxi, int& num)
{
  Complex* tmp=(Complex*)malloc(sizeof(Complex)*(num+1));
  for(int i=0;i<num;i++)
  {
    new (tmp+i) Complex(dev_minC[i].r, dev_minC[i].i);
  }
  new (tmp+num) Complex(minr, mini);
  for(int i=0;i<num;i++)
  {
    dev_minC[i].~Complex();
  }
  free(dev_minC);
  dev_minC=tmp;

  tmp=(Complex*)malloc(sizeof(Complex)*(num+1));
  for(int i=0;i<num;i++)
  {
    new (tmp+i) Complex(dev_maxC[i].r, dev_maxC[i].i);
  }
  new (tmp+num) Complex(maxr, maxi);
  for(int i=0;i<num;i++)
  {
    dev_maxC[i].~Complex();
  }
  free(dev_maxC);
  dev_maxC=tmp;

  min_r = & dev_minC[num].r;
  min_i = & dev_minC[num].i;
  max_r = & dev_maxC[num].r;
  max_i = & dev_maxC[num].i;

  num++;
    
}

void RmRange(int& num)
{
  if(num>1)
  {
    Complex* tmp=(Complex*)malloc(sizeof(Complex)*(num-1));
    for(int i=0;i<num-1;i++)
    {
      new (tmp+i) Complex(dev_minC[i].r, dev_minC[i].i);
    }
    for(int i=0;i<num;i++)
    {
      dev_minC[i].~Complex();
    }
    free(dev_minC);
    dev_minC=tmp;

    tmp=(Complex*)malloc(sizeof(Complex)*(num-1));
    for(int i=0;i<num-1;i++)
    {
      new (tmp+i) Complex(dev_maxC[i].r, dev_maxC[i].i);
    }
    for(int i=0;i<num;i++)
    {
      dev_maxC[i].~Complex();
    }
    free(dev_maxC);
    dev_maxC=tmp;

    min_r = & dev_minC[num-2].r;
    min_i = & dev_minC[num-2].i;
    max_r = & dev_maxC[num-2].r;
    max_i = & dev_maxC[num-2].i;

    num--;
  }
}

void CudaCalcu()
{
    int* d_ifMBSet;
    int* d_itera;
    int* N;
    cudaMalloc((void **)&N, sizeof(int));
    cudaMalloc((void **)&d_ifMBSet, pixels * sizeof(int));
    cudaMalloc((void **)&d_itera, pixels * sizeof(int));

    float* d_min_r;
    float* d_min_i;
    float* d_max_r;
    float* d_max_i;
    cudaMalloc((void **)&d_min_r, sizeof(float));
    cudaMalloc((void **)&d_min_i, sizeof(float));
    cudaMalloc((void **)&d_max_r, sizeof(float));
    cudaMalloc((void **)&d_max_i, sizeof(float));

    //----------mem copy-------------

    cudaMemcpy(N, p, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min_r, min_r, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_min_i, min_i, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_r, max_r, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_i, max_i, sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blocks(64, 128);
    dim3 threads(4, 8);
    isMBpoint<<<blocks, threads>>>(d_min_r, d_min_i, d_max_r, d_max_i, N, d_ifMBSet, d_itera);

    cudaMemcpy(ifMBSet, d_ifMBSet, pixels * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(itera, d_itera, pixels * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_ifMBSet);
    cudaFree(d_itera);
    cudaFree(N);
    cudaFree(d_min_r);
    cudaFree(d_min_i);
    cudaFree(d_max_r);
    cudaFree(d_max_i);
}
void CalImage()
{
  if(image==NULL)
  {
    image = new GLfloat[pixels*3];
  }
  for(int i=0;i<pixels;i++)
  {
    int px = i/window;
    int py = i%window;
    int id = (py)*window+(px);
    if(ifMBSet[i]==0)
    {
      image[id*3] = colors[itera[i]-1].r;
      image[id*3+1] = colors[itera[i]-1].g;
      image[id*3+2] = colors[itera[i]-1].b;
    }
    else
    {
      image[id*3] = 0.0;
      image[id*3+1] = 0.0;
      image[id*3+2] = 0.0;
    }
    
  }
  if(image2==NULL)
  {
    image2 = new GLfloat[pixels*3];
  }
  for(int i=0;i<pixels;i++)
  {
      image2[i*3]=image[i*3];
      image2[i*3+1]=image[i*3+1];
      image2[i*3+2]=image[i*3+2];

  }

}
//-------------------------------------------------------------
//-------------------imitation-------------------

void init()
{
  //select clearing (background) color
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glShadeModel(GL_FLAT);
}


void reshape(int w, int h)
{
  glViewport(0,0, (GLsizei)w, (GLsizei)h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, (GLdouble)w, (GLdouble)0.0, h, (GLdouble)-w, (GLdouble)w);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}


void timer(int)
{
  glutPostRedisplay();
  glutTimerFunc(1000.0 / updateRate, timer, 0);
}
//-------------------------------------------------
void showMandelbrot()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(-1.0,1.0,-1.0,1.0);   //x1 x2    y1 y2

  glDrawPixels(w, w, GL_RGB, GL_FLOAT, image2);

  glutSwapBuffers();
}


void Keyboard(unsigned char key, int x, int y)
{
    if (key == 'b')
    {
      RmRange(num);
      CudaCalcu();
      cout<<num<<"#"<<endl<<endl<<endl;
      CalImage();
    }
}

int px1,py1,px2,py2;
float xx1,yy1,xx2,yy2;
int button_down=0;
void Mouse(int button, int state, int x, int y)
{
    if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) 
    {
 //     cout<<"DOWN"<<x<<","<<y<<endl;
      px1=x;
      py1=y;
      button_down=1;
    }
    if(button == GLUT_LEFT_BUTTON && state == GLUT_UP) 
    {
 //     cout<<"UP"<<x<<","<<y<<endl;
      px2=x;
      py2=y;
      button_down=0;

   
      int widthx=0; 
      int widthy=0;
      int width=0;
      if(px1>px2)
      {
        widthx=px1-px2;
      }
      else{
        widthx=px2-px1;
      }
      if(py1<py2)
      {
        widthy=py2-py1;
      }
      else{
        widthy=py1-py2;
      }

      if(widthx<widthy){
        width=widthx;
      }
      else{
        width=widthy;
      }
      
      if(px1>px2)
      {
        px2=px1-width;  
        xx1=dev_minC[num-1].r + (dev_maxC[num-1].r-dev_minC[num-1].r)*(px2*1.0/window);
        xx2=dev_minC[num-1].r + (dev_maxC[num-1].r-dev_minC[num-1].r)*(px1*1.0/window);
      }
      else{
        px2=px1+width;
        xx1=dev_minC[num-1].r + (dev_maxC[num-1].r-dev_minC[num-1].r)*(px1*1.0/window);
        xx2=dev_minC[num-1].r + (dev_maxC[num-1].r-dev_minC[num-1].r)*(px2*1.0/window);
      }
      if(py1<py2)
      {
        py2=py1+width;
        yy1=dev_maxC[num-1].i - (dev_maxC[num-1].i-dev_minC[num-1].i)*(py2*1.0/window);
        yy2=dev_maxC[num-1].i - (dev_maxC[num-1].i-dev_minC[num-1].i)*(py1*1.0/window);

      }
      else{
        py2=py1-width;
        yy1=dev_maxC[num-1].i - (dev_maxC[num-1].i-dev_minC[num-1].i)*(py1*1.0/window);
        yy2=dev_maxC[num-1].i - (dev_maxC[num-1].i-dev_minC[num-1].i)*(py2*1.0/window);

      }

      AddRange(xx1, yy1, xx2, yy2,num);
      CudaCalcu();
  //    cout<<num<<"#"<<endl<<endl<<endl;
      CalImage();
      glutDisplayFunc(showMandelbrot);
    }
}

void OnMouseMove(int x, int y)          /*当鼠标移动时会回调该函数*/

{
    if(button_down)         /*如果鼠标没有按下则不改变摄像机位置*/

    {
      for(int i=0;i<pixels;i++)
      {
        image2[i*3]=image[i*3];
        image2[i*3+1]=image[i*3+1];
        image2[i*3+2]=image[i*3+2];
      }
      if(x<px1){
        for(int i=x;i<=px1;i++)
        {
          
          image2[((window-y)*window+i)*3]=0;
          image2[((window-y)*window+i)*3+1]=0;
          image2[((window-y)*window+i)*3+2]=1;
          image2[((window-y-1)*window+i)*3]=0;
          image2[((window-y-1)*window+i)*3+1]=0;
          image2[((window-y-1)*window+i)*3+2]=1;

          image2[((window-py1)*window+i)*3]=0;
          image2[((window-py1)*window+i)*3+1]=0;
          image2[((window-py1)*window+i)*3+2]=1;
          image2[((window-py1-1)*window+i)*3]=0;
          image2[((window-py1-1)*window+i)*3+1]=0;
          image2[((window-py1-1)*window+i)*3+2]=1;
         }

      }
      else
      {
          for(int i=px1;i<=x;i++)
        {
          
          image2[((window-y)*window+i)*3]=0;
          image2[((window-y)*window+i)*3+1]=0;
          image2[((window-y)*window+i)*3+2]=1;
          image2[((window-y-1)*window+i)*3]=0;
          image2[((window-y-1)*window+i)*3+1]=0;
          image2[((window-y-1)*window+i)*3+2]=1;

          image2[((window-py1)*window+i)*3]=0;
          image2[((window-py1)*window+i)*3+1]=0;
          image2[((window-py1)*window+i)*3+2]=1;
          image2[((window-py1-1)*window+i)*3]=0;
          image2[((window-py1-1)*window+i)*3+1]=0;
          image2[((window-py1-1)*window+i)*3+2]=1;
        }

      }

      if(y<py1){

        for(int i=y;i<=py1;i++){
          
          image2[((window-i)*window+x)*3]=0;
          image2[((window-i)*window+x)*3+1]=0;
          image2[((window-i)*window+x)*3+2]=1;
          image2[((window-i)*window+x+1)*3]=0;
          image2[((window-i)*window+x+1)*3+1]=0;
          image2[((window-i)*window+x+1)*3+2]=1;

          image2[((window-i)*window+px1)*3]=0;
          image2[((window-i)*window+px1)*3+1]=0;
          image2[((window-i)*window+px1)*3+2]=1;
          image2[((window-i)*window+px1+1)*3]=0;
          image2[((window-i)*window+px1+1)*3+1]=0;
          image2[((window-i)*window+px1+1)*3+2]=1;
        }
      }
      else
      {
        for(int i=py1;i<=y;i++){
          image2[((window-i)*window+x)*3]=0;
          image2[((window-i)*window+x)*3+1]=0;
          image2[((window-i)*window+x)*3+2]=1;
          image2[((window-i)*window+x+1)*3]=0;
          image2[((window-i)*window+x+1)*3+1]=0;
          image2[((window-i)*window+x+1)*3+2]=1;

          image2[((window-i)*window+px1)*3]=0;
          image2[((window-i)*window+px1)*3+1]=0;
          image2[((window-i)*window+px1)*3+2]=1;
          image2[((window-i)*window+px1+1)*3]=0;
          image2[((window-i)*window+px1+1)*3+1]=0;
          image2[((window-i)*window+px1+1)*3+2]=1;
        }
      }
    }

}

//--------------------------------------------------


int main(int argc, char** argv)
{
  InitializeColors();

  // Set up necessary host and device buffers
  for(int i=0;i<2001;i++)
  {
//    cout<<"("<<colors[i].r<<","<<colors[i].g<<","<<colors[i].b<<")"<<endl;
  }

  ifMBSet = new int[pixels];
  itera = new int[pixels];
  for(int i=0;i<pixels;i++)
  {
    ifMBSet[i] = 0;
    itera[i] = 0;
  }
 
  //---------------------------------------------------

  AddRange(minC.r, minC.i, maxC.r, maxC.i,num);
  CudaCalcu();
//  cout<<num<<"#"<<endl<<endl<<endl;
  CalImage();
  

  // Initialize OPENGL here
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(512, 512);
  glutInitWindowPosition(100, 100);
  glutCreateWindow("Mandelbrot");
  init();

  glutDisplayFunc(showMandelbrot);
  glutKeyboardFunc(Keyboard);
  glutMouseFunc(Mouse);  
  glutMotionFunc(OnMouseMove); 
 
  glutReshapeFunc(reshape);
  updateRate=10;
  glutTimerFunc(1000.0 / updateRate, timer, 0);

  glutMainLoop(); // THis will callback the display, keyboard and mouse


  delete [] ifMBSet;
  delete [] itera;
  delete [] colors;

  return 0;
   
}
