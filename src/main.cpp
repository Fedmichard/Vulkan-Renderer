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

// A seperate function for actually creating the Debug Messenger
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    // vkGetInstanceProcAddr will return nullptr if the function couldn't be loaded because we are looking for the address of vkCreateDebugUtilsMessengerEXT
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

// Once again since our Debug Messenger is a vulkan extension we have to find the function that will delete our messenger instance
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    // vkDestroyDebugUtilsMessengerEXT will return nullptr if the function couldn't be loaded because we are looking for the address of vkDestroyDebugUtilsMessengerEXT
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

class HelloTriangleApplication{
private:
    // GLFW window instance
    GLFWwindow* window;

    // First thing to do when initializing vulkan library is to create an instance
    // The instance is the connection between your app and the vulkan library
    VkInstance instance;

    // Managed with a callback that needs to be explicitly created and destroyed
    VkDebugUtilsMessengerEXT debugMessenger;

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

    // Setting up a callback function to handle messages and its associated details
    // Will return a vector of c-style string literals
    // The extensions specified by GLFW are always going to be required
    std::vector<const char*> getRequiredExtensions() {
        // GLFW has a function that returns extensions it needs to do that which we can pass to the struct
        // Next layers specify the global extensions
        // retrieve count of available glfw extensions
        uint32_t glfwExtensionCount = 0;
        // return the vulkan instance extensions that glfw require
        // Points to an array of c-style strings
        const char** glfwExtensions;
        // Vulkan is a platform agnostic API, which means that you need an extension to interface with the window system
        // glfwGetRequiredInstanceExtensions returns an array of c-style strings
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        // Range constructor of std::vector
        // Glfw is pointer that points to the start of an array of c-style string literals
        // Points to the beginning of glfwExtensions array and 1 element past it
        // Extension count is updated by glfwGetRequiredInstanceExtensions to return a number
        // since we start count at 0 and not 1 that number is going to be 1 over
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        // If valdiation layers are enabled
        if (enableValidationLayers) {
            // VK_EXT_DEBUG_UTILS_EXTENSION_NAME is a macro that is literal string "VK_EXT_debug_utils"
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
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
            bool layerFound = false;
            // For Each validation layer
            for (const auto &layerProperties : availableValidationLayers) {
                // For each validation layer available on our system 
                // Compare the strings of their name (since their names are unique)
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    // If our requested validation layer exists, return true
                    layerFound = true;
                    break;
                }
            }

            // If the layer isn't found return false
            if (!layerFound) {
                return false;
            }
        }

        return true;
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

    /*
        Static meaning it's only visible within the current source file it is defined (can't be accessed by other cpp files)
        VKAPI_ATTR Essentially ensures that the function is exported corectly for the vulkan api to call it
        VkBool32 is the return type it is a bool value of 32 bits (0 or 1)
        VKAPI_CALL is another predefined macro it specifies the calling convention the function will use
        Ensures vulkan call back functions are called correctly
        The vulkan validation layers and driver will call this function whenver they have a message (error, warning, informational)
    */
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageServerity,
        VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData) {
            /*
                VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity indicates message severity (verbose, info, warning, error)
                Message severity is setup in a way that you can check which one is worse then one another (the enums have # values)
                VkDebugUtilsMessageTypeFlagsEXT messageType indicates category of message (general, validation - violates specification or possible mistake, performance - potentially non optimal use of vulkan)
                const VkDebugUtilsMessengerCallbackDataFlagsEXT* pCallBackData a pointer to struct containing detailed info about message
                Important members are:
                    pMessage: The debug message as a null-terminated string
                    pObjects: Array of Vulkan object handles related to the message
                    objectCount: Number of Objects in array
                Void* pUserData user defined pointer that you can specify when setting up the debug messenger. Allows you to pass custom data to callback
            */

            std::cerr << "Validation Layer: " << pCallbackData->pMessage << std::endl;

            return VK_FALSE;
    }

    // The general pattern that object creation function params in vulkan follow is:
    // Pointer to struct with creation info
    // Pointer to custom allocator callbacks
    // Pointer to the varaible that stores the handle to the new object
    void createInstance() {
        // If validation layers are enabled (non debug mode) and validation layer support is false
        // List all available validation layers on this system
        if (enableValidationLayers && !checkVulkanInstanceLayers()) {
            throw std::runtime_error("Validation layers requested, but not available!");
        }

        // Fill a struct with some information
        // Technically optional, but it may provide some useful information to the driver in order to optimize our specific app
        // A lot of info in vulkan is passed through structs instead of function params
        VkApplicationInfo appInfo{}; // set all of the default values of appInfo
        // Many structs require you to explicitly specify the type
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; // structure type is an application info type, Has a list of VkStructureType enums
        appInfo.pApplicationName = "Hello Triangle"; // name of our application
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0); // it's version
        appInfo.pEngineName = "No Engine"; // The name of our engine
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0); // the version of our engine
        appInfo.apiVersion = VK_API_VERSION_1_0; // The api version

        // We'll have to fill in one more struct to provide sufficient info for creating an instance
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        // Reusing our function in create instance
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        // Determine the global validation layers to enable
        // if enable validation layers is true
        if (enableValidationLayers) {
            // We'll have 1 enabled layer count
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            // which validation layers we'll have enabled
            createInfo.ppEnabledLayerNames = validationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0; 

            createInfo.pNext = nullptr;
        }

        // List all available vulkan instance extensions on this system
        checkVulkanInstanceExtensions();

        // Now specified everything Vulkan needs to create an instance and we can finally issue the vkCreateInstance call
        // Creating a VKInstance object initializes the vulkan library and allows the app to pass info about itself to the implementation
        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

        // Check to see if our Vulkan instance was successfully created
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create an instance!");
        }
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        // All the message severity's I want my callback to be called for
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        // All the messages I want my callback to be notified for (enabled all of them)
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData = nullptr; // optional
    }

    void setUpDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        // This function is not automatically loaded because it is an extension function
        // The debug messenger is specific to our Vulkan instance and its layers, it needs to be explicitly specified as first arugment
        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("Failed to set up debug messenger!");
        }
    }

    // Store and initiate each vulkan object
    void initVulkan() {
        // Is called on it's own to initialize vulkan
        createInstance();
        setUpDebugMessenger();
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
        if (enableValidationLayers) {
            // DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }
        // Destroy vkinstance right before the program exits
        vkDestroyInstance(instance, nullptr);

        // Once the window is closed we need to clean up resources by destroying it
        glfwDestroyWindow(window);

        // Terminate GLFWs
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