# High-Leverage Computing Library

## What it does
It mimics the api style of famous Eigen library , and computes using multiple backend , such as Kompute and AdaptiveCPP

## Is it necessary
Yes it does , the community lack of this type of library using gpu backend and also compatible with cpu backend .
Even those who could do all above , they usually require custom toolchain , unlike sycl or Kompute , they use regular toolchain .

## Why not just use AdaptiveCPP
Because AdaptiveCPP uses clspv toolchain , which is not easy to use and embed.
Also I was in charge of AdaptiveCPP pack in xmake and I temporarily have no time to finish clspv toolchain integration . 
If you wish to fully use AdaptiveCPP , it's welcome to pr to xmake-repo and here!

## How can we sure about that your project is not a tool
For one who is very serious about correctness , I build the whole test pipeline , and support coverage computing . 
