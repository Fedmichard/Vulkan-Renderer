/* GUIDE FOR LATER
First draw a triangle.
Then a quad.
Then a cube.
Then an army of cubes.
Then do the army of cubes with deferred rendering, forward+ rendering.
Do some sort of occlusion culling.
Implement picking.
*/

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>
#include <stdexcept> //Provides Try Catch Logic
// provides EXIT_SUCCESS and EXIT_FAILURE macros
#include <cstdlib>
#include <cstring>

// unsigned 32 bit ints for the width and the height, doesn't matter just means can't be negative
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// After the vector is initialized you cannot modify the vector itself
// It's an array of C-style string literals, text in quotation marks that are unchangeable
// VK_LAYER_KHRONOS_validation is the standard library included in lunarg vulkan SDK
// A long list of individual validation layers
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

// NDEBUG macro is apart of C++ standard and it just means if the program is being compiled in debug mode or not
#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else  
    const bool enableValidationLayers = true;
#endif

class HelloTriangleApplication{
private:
    // GLFW window instance
    GLFWwindow* window;

    // First thing to do when initializing vulkan library is to create an instance
    // The instance is the connection between your app and the vulkan library
    VkInstance instance;

    // Initialize GLFW and create a window
    void initWindow() {
        // Initializes glfw library, but it was originally designed for an OpenGL context
        // GLFW will default to OpenGL
        glfwInit();

        // To prevent it to not create an OpenGL context we write this
        // We are explicitly disabling OpenGL context creation
        // Hint, API: Used to specify which graphics API the window will be created for, glfw will not initialize OpenGL
        // Responsible for creating and managing a Vulkan instance
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        
        // Prevent window resizing
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        // Create GLFW window
        // WIDTH, HEIGHT, "WINDOW NAME", Specify which monitor, OpenGL specific
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

    }

    // Function that checks if all of the requested layers are available
    // Same process for checking if requested instance extensions are available
    bool checkVulkanInstanceLayers() {
        // unsigned 32 bit int used to retrieve count of available layers
        uint32_t layerCount = 0;

        // updates our layer count based on what is available on our system
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        // Vector to hold all the available vkInstanceLayerProperties
        std::vector<VkLayerProperties> availableValidationLayers(layerCount);
        // Then we'll run it again to actually update our layer vector
        vkEnumerateInstanceLayerProperties(&layerCount, availableValidationLayers.data());

        // Same loop
        std::cout << layerCount << " Available layers:" << std::endl;

        for (const auto &layer : availableValidationLayers) {
            std::cout << "\t" << layer.layerName << std::endl;
        }

        // Now we're going to check if all the layers in validationLayers exist in our availableValidationLayers vector
        for (const char* layerName : validationLayers) {
            
        }

        return false;
    }

    void checkVulkanInstanceExtensions() {
        // Sometimes while running vkCreateInstance you may get an error code that is VK_ERROR_EXTENSION_NOT_PRESENT
        // This basically means that an extension we require isn't available, we could specify the specific extension we require but
        // If we want to check for optional functionality we can retrieve a list of supported extensions before creating an instance
        uint32_t extensionCount = 0;

        // You can retrieve a list of supported extensions using this function
        // 3rd param takes a pointer to a variable that stores the nubmer of extensions and a vkExtensionProperties to store details of the extensions
        // We can leave this param empty though to request the number of extensions
        // The first param is optional that allows us to filter extensions by a specific validation layer
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        // Allocate an array to hold the extension details
        std::vector<VkExtensionProperties> extensions(extensionCount);
        // Then we can query the extension details:
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        std::cout << extensionCount << " Available extensions:" << std::endl;

        // Loop through all our extensions and list them
        for (const auto &extension : extensions) {
            std::cout << "\t" << extension.extensionName << std::endl;
        }

    }

    // The general pattern that object creation function params in vulkan follow is:
    // Pointer to struct with creation info
    // Pointer to custom allocator callbacks
    // Pointer to the varaible that stores the handle to the new object
    void createInstance() {
        // Fill a struct with some information
        // Technically optional, but it may provide some useful information to the driver in order to optimize our specific app
        // A lot of info in vulkan is passed through structs instead of function params
        VkApplicationInfo appInfo{};
        // Many structs require you to explicitly specify the type
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        // We'll have to fill in one more struct to provide sufficient info for creating an instance
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        // GLFW has a function that returns extensions it needs to do that which we can pass to the struct
        // Next layers specify the global extensions
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        // Vulkan is a platform agnostic API, which means that you need an extension to interface with the window system
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;
        // Determine the global validation layers to enable
        // Leave these empty for now
        createInfo.enabledLayerCount = 0;

        // Now specified everything Vulkan needs to create an instance and we can finally issue the vkCreateInstance call
        // Creating a VKInstance object initializes the vulkan library and allows the app to pass info about itself to the implementation
        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

        // List all available vulkan instance extensions on this system
        checkVulkanInstanceExtensions();

        // List all available validation layers on this system
        checkVulkanInstanceLayers();
        
        // Check to see if our Vulkan instance was successfully created
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create an instance!");
        }
    }

    // Store and initiate each vulkan object
    void initVulkan() {
        // Is called on it's own to initialize vulkan
        createInstance();
    }

    // Loop that iterates until the window is closed in a moment
    void mainLoop() {
        // Keep the application running until an error occurs or the window is closed
        // While window is open
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents(); // Checks for events 
        }
    }

    // Once window is is closed we'll deallocate used resources
    // Every Vulkan object that we create needs to be destroyed when we no longer need it
    // It is possible to perform automatic resource management using RAII or smart pointers
    void cleanup() {
        // Destroy vkinstance right before the program exits
        vkDestroyInstance(instance, nullptr);

        // Once the window is closed we need to clean up resources by destroying it
        glfwDestroyWindow(window);

        // Terminate GLFW
        glfwTerminate();
    }

public:
    // Our actual application will be ran through this
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}