################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Sources/CVtoYarp.cpp \
../Sources/saliencyDemo.cpp 

OBJS += \
./Sources/CVtoYarp.o \
./Sources/saliencyDemo.o 

CPP_DEPS += \
./Sources/CVtoYarp.d \
./Sources/saliencyDemo.d 


# Each subdirectory must supply rules for building sources it contributes
Sources/%.o: ../Sources/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/home/icub/new_opencv/opencv/include -I/usr/local/include/yarp -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


