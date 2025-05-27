/* GUIDE FOR LATER
First draw a triangle.
Draw Several Triangles.
Add Movement to triangles.
Then a quad.
Render 2 triangles and order the verices in a way so that you have a plane
Make the plane have some albedo texture, like wood
Implement basic phong shading model
Add normal texture
Anti-Alias the scene
(Optional) Abstract the texture management to some simple material class
Try load 3D model and do steps 2-4 again
Then a cube.
Then an army of cubes.
Then do the army of cubes with deferred rendering, forward+ rendering.
Do some sort of occlusion culling.
Implement picking.
*/
// #define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <iostream>
#include <fstream>
#include <optional>
#include <array>
#include <vector>
#include <map>
#include <limits>
#include <algorithm>
#include <set>
#include <stdexcept> //Provides Try Catch Logic
// provides EXIT_SUCCESS and EXIT_FAILURE macros
#include <cstdlib>
#include <cstring>

/*
    how many frames should be allowed to process concurrently
    The cpu and gpu don't run in parallel so the cpu may just write new commands before the gpu has time to finish executing it
    to combat this we use a fence to signal the host when a task has finished executing on the gpu

    right now we are waiting for the previous frame to finish before rendering to a new one
    this leads to unnecessary idling of the host which means there will be a period where the CPU won't be doing anything
    this controls the number of frames the cpu can prepare ahead of the gpu at most, how many frames can be processed at any given time

    we don't want our cpu to get too far ahead of the GPU, if the cpu finishes early it'll just go back to that idle but it's still much more efficient
*/
const int MAX_FRAMES_IN_FLIGHT = 2;
uint32_t currentFrame = 0;

// unsigned 32 bit ints for the width and the height, doesn't matter just means can't be negative
const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// After the vector is initialized you cannot modify the vector itself
// It's an array of C-style string literals, text in quotation marks that are unchangeable
// VK_LAYER_KHRONOS_validation is the standard library included in lunarg vulkan SDK
// This vector is long list of individual validation layers
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation" // Right now we're only getting this one validation layer for tht catches common vulkan usage errors
};

// List of required device extensions for compatible device
const std::vector<const char*> deviceExtensions = {
    // Is a device extension since it relies on windowing system
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

// NDEBUG macro is apart of C++ standard and it just means if the program is being compiled in debug mode or not
#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else  
    const bool enableValidationLayers = true;
#endif

// A seperate function for actually creating the Debug Messenger
// The debug messenger will issue a message to the debug callback when an event of interest occurs
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    // vkGetInstanceProcAddr will return nullptr if the function couldn't be loaded because we are looking for the address of vkCreateDebugUtilsMessengerEXT
    // Essentially a loader function that is crucial for using extension functions in vulkan
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

/*
    Bundles all of our queue family indices
    Each queue family on physical device is given a unique index (unint32_t)
    Vulkan is designed for parallelism, by having multiple queuees you can submit different types of work to the gpu concurrently
    Queue families perform operations, they're just groups of operations specific to one thing
*/
struct QueueFamilyIndices {
    /*
        The first queue family we're going to need is the graphicsFamily queue family represented by some uint32_t that we'll define later
        We can use the std::optional wrapper which contains no value until we assign something to it
        We can query it at any point using .has_value() function
        This is just to help dictate whether or not this queue family was available(found)
        This is necessary because sometimes we may prefer devices but not necessarily require it
        receives graphics commands (like drawing)
    */
    std::optional<uint32_t> graphicsFamily;

    /*
        We're going to add another queue for the presentation of images since they could not overlap depending on your device
        The present family is a queue on the logical device that can perform presentation operations
        You can use this queue to place commands that tell vulkan to display an image from the swap chain on the screen
    */
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() &&
                presentFamily.has_value();
    }
};


/*
    Just checking if the swapchain is available isn't enough
    It may be available but incompatible with our window surface
    It also involves a lot more settings than creating an instance or a device
    There are 3 kinds of properties we need to check
*/
struct SwapChainSupportDetails {
    // structure that defines capabilities of a surface
    VkSurfaceCapabilitiesKHR capabilities;
    // vector that holds surface formats
    std::vector<VkSurfaceFormatKHR> formats;
    // vector that holds available presentation nodes
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
    // our actual vertex in a 2d space
    // always need this
    glm::vec2 pos;
    // an attribute associated with our vertex
    // this is the vertices color attribute
    glm::vec3 color;

    /*
        the next step would be telling vulkan how to pass this data format to the vertex data when we pass it into gpu memory
        a vertex binding describes the rate to load data from memory throughout vertices

        specifies the number of bytes between data entries and whether to move to next data entry after each vertex or each instance
        can think of it as one stream or layout of vertex data and how it's organized in memory for retrieval and processing
        defines overall layout of one chunk of vertex data, being this one struct

        more about the structure of the data scheme
    */
    static VkVertexInputBindingDescription bindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        // specifies the index of the binding in the array of bindings
        // we only have one binding so it's 0 (the first one)
        bindingDescription.binding = 0;
        // number of bytes from one entry to the next
        bindingDescription.stride = sizeof(Vertex);
        // VK_VERTEX_INPUT_RATE_VERTEX move on to the next data entry after each vertex
        // VK_VERTEX_INPUT_RATE_INSTANCE move on to the next data entry after each instance
        // so whether we're advancing per vertex or per instace
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    // we're creating a fixed size array of 2 VkVertexInputAttributeDescriptions
    // input into you're shader, describes how an individual vertex attribute is extracted from a chunk of vertex data originating from a specific binding description
    // Since we have 2 attributes being position and color, we'll have an array of 2 attributes which will define how each are extracted
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        // how we extract position attribute
        // binding is the unique identifier that connects description to vertex buffer
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        // its represents 2 32 bit integer values (pos.x, pos.y)
        // aka a vec2
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        // how we extract color attribute
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        // aka a vec3
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

class HelloTriangleApplication{
private:
    // GLFW window instance
    GLFWwindow* window;

    // First thing to do when initializing vulkan library is to create an instance
    // The instance is the connection between your app and the vulkan library
    VkInstance instance;
    // Managed with a callback that needs to be explicitly created and destroyed
    VkDebugUtilsMessengerEXT debugMessenger;
    // Window surface
    // A vulkan object that represents the rendering target for your vulkan commands
    // Cannot directly render to glfw window using vulkan commands you need this surface to link to that window and write to the surface
    VkSurfaceKHR surface;

    // Will be implicitly destroyed when VkInstance is destroyed
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    // Logical device
    VkDevice device;

    // Queues
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    // our swapChain
    VkSwapchainKHR swapChain;
    // Will store the handles of the VkImages
    // We'll reference these during rendering operations
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    // To use any VkImage including those in the swap chain in the render pipeline we need to create a VkImageView object
    // An image view is quite literally a view into an image, it describes how to access the image and which part of the image to access
    std::vector<VkImageView> swapChainImageViews; // We'll leave it as a basic image view for every image in the swap chain so we can use them as color targets
    // now we can create the frame buffers so we can specify the attachments expected by the render pass during creation
    // our render pass will expect a single framebuffer with the same format as a swap chain image
    std::vector<VkFramebuffer> swapChainFramebuffers;

    // our renderpass
    VkRenderPass renderPass;

    // We must create a pipeline layout even though we won't be using it right now, we'll just leave it empty
    // We can pass uniform values in shaders, which are global values that can be accessed and changed at drawtime through our entire pipeline without the need for recompiling
    // They are most common for passing transformation matrix to the vertex shader or to create texture samplers
    VkPipelineLayout pipelineLayout;

    // our pipeline
    VkPipeline graphicsPipeline;

    // our command pool
    VkCommandPool commandPool;

    // our vertex buffer
    // logical step
    VkBuffer vertexBuffer; 
    // handle to the actual physically allocated block of GPU memory, this represents a chunk of memory on the GPU
    // this is needed so we can bind our buffer to an actual
    VkDeviceMemory vertexBufferMemory;

    /*
        So the memory type that we currently have linked to our vertexbuffer on our gpu may not be the most compatible type
        the memory type that we picked and is compatible with our vertex buffer has either the VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT flag
        while this allows us the map and memcpy from the cpu into the gpu, it isn't the most optimal location and flag for the gpu to read from (because we pass our vertex buffer that has an index to the heap on gpu)
        
        the most optimal memory type has the VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT flag but usually the CPU can't access it on dedicated graphics cards 
        that's why we're going to write to our staging buffer, move it to the vertex buffer, and read from the vertex buffer so we get the bost of both worlds

        we will need a transfer queue so we can get the buffer copy command from one buffer to another
    */
    VkBuffer stagingBuffer;

    // EACH FRAME SHOULD HAVE IT'S OWN COMMAND BUFFERS, SEMAPHORES, AND FENCES; SINCE WE ARE GONNA HAVE 2 IN FLIGHT FRAMES AT A TIME
    // our command buffer
    std::vector<VkCommandBuffer> commandBuffers;
    // semaphores to be used on the gpu side to signal that a rendering operation has finished executing and that an image has been acquired and available
    // sync point between gpu to gpu operations
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    // fence to make sure only frame is being rendered at a time on the cpu side
    // sync point between cpu and gpu operations
    std::vector<VkFence> inFlightFences;

    // this is going to be used to handle resizes explicitly 
    // we'll use this to flag when a resize has occured
    bool framebufferResized = false;

    // our vectors
    const std::vector<Vertex> vertices = {
        // Position     // Color
        {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
        {{0.5f, 0.5f},  {0.0f, 1.0f, 0.0f}},
        {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    };

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
        // glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        // Create GLFW window
        // WIDTH, HEIGHT, "WINDOW NAME", Specify which monitor, OpenGL specific
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        // we do this because GLFW is a C library and doesn't handle the concept of this from classes
        // so we're passing a pointer of our HelloTriangleApplication class 
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

    }

    // we're creating a static function as a callback because GLFW doesn't know how to properly call a member function with the correct "this" pointer 
    // static members belong to the class themselves and not a specific instance
    static void framebufferResizeCallback(GLFWwindow *window, int width, int height) {
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    // Setting up a callback function to handle messages and its associated details
    // Will return a vector of c-style string literals
    // The extensions specified by GLFW are always going to be required
    // GLFW is not a vulkan extension so we have to retrieve it like this so that we may use it with our vulkan app
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

    // Information about for our debug messenger
    // The kind of message severity's I want to be detected as well as the types of messages
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

    // Checking if each device is suitable for our needs
    bool isDeviceSuitable(VkPhysicalDevice device) {
        /*
        // To evaluate the suitability of a device we can start by querying for some details
        // To query basic device properties (name, type, and supported vulkan version) can be queried using vkGetPhysicalDeviceProperties
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        // To query for optional features like texture compression, 64 bit floats and multi viewport rendering (for vr) can be queried with this
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
        */

        // Finds our desired queue families and ensures it is compatible
        QueueFamilyIndices indices = findQueueFamilies(device);

        // Check if extensions are supported on physical device
        bool extensionsSupported = checkDeviceExtensionSupport(device);

        // Check if swap chain support is adequate
        // Right now our only device extension is swap chain so we're check if swap chain is supported first
        bool swapChainAdequate = false;
        // So if all extensions are supported
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        /*
            What we do here is we first collect all of the available device extensions compatible with our physical device
            Just like how we found the compatible vulkan instance extensions and validation layers compatible with our gpu
            We then create a set of our required device extensions so that we don't affect the original array of extensions
            we iterate through the available device extensions and as we do we erase that string from our required extension set
            if at the end of the loop the extension is erased, that means it's compatible
        */
        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }
    
    /*
        Any operation done using vulkan needs to be submitted to a queue
        There are different kinds of queues from different queue families
        Each family of queue only allows a subset of commands like one family may only process compute commands
        We need to find out all the queue families that are compatible with our selected physical device
        Used to be a uint32_t
    */
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        // Right now we're only going to look for a queue that supports graphics commands
        QueueFamilyIndices indices;

        // We're looking for all the queue families available on my system
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        // Has some info such as the type of operations supported and the number of queues that can be created
        // With the queue families from our device we're looking for all the ones that support VK_QUEUE_GRAPHICS_BIT so we can add to our indicies perhaps?
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport) {
                indices.presentFamily = i;
            }

            // We need to find at least one queue family that supports VK_QUEUE_GRAPHICS_BIT
            // If the queue family the index is currently on supports VK_QUEUE_GRAPHICS_BIT we will set our graphicsFamily index to that
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }

        return indices;
    }

    /*
        Finding the right settings for the best possible swap chain
        First thing we need is the surface format

        A surface format defines how the pixels of an image in a swapchain are represented in memory
        We'll pass a reference to our formats vector member of our SwapChainSupportDetails struct

        Each VkSurfaceFormatKHR has a format-color space pair
        The format member specifies the color channels and types (like rgb and alpha channels)
        The color space member indicates indiciates if the SRGB color space is supported or not
        If SRGB is available we'll use it because it provides more accurate perceived colors
        It is also the standard colorspace 
    */
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
        // First let's see if prefered combination is available to use
        for (const auto& availableFormat : availableFormats) {
            // Go through all of our available fromats
            // If both of these options are available from our available formats return that available format
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                /*
                    VK_FORMAT_B8G8R8A8_SRGB means we store BGRA in that order as 8 bits each for a total of 32 bits per pixel
                    Each pixel in our swapchain images will be stored as 32 bits in that order in memory
                    VK_COLOR_SPACE_SRGB_NONLINEAR_KHR is a flag used to tell us if SRGB color space is supported or not
                */
                return availableFormat;
            }

            // If that fails meaning we lack the support of both, we can usually just select the first one
            // Or we can order them by how "good" they are
            return availableFormats[0];
        }
    }

    /*
        Next thing we need is presentation mode which is arguably the most important setting for the swap chain
        It represents the actual conditions to show images on the screen
        This part actually helps synchronize the displays refresh rate to your renders

        There are 4 possible modes available: The moment the display is refreshed is known as vertical blank
        VK_PRESENT_MODE_IMMEDIATE_KHR: Images submitted by app are transferred to the screen immediately which may result in tearing
        VK_PRESENT_MODE_FIFO_KHR: Swapchain becomes a queue(FIFO) where the display takes an image from the front of the queue when the display
                                  is refreshed and the program inserts rendered images at the back of queue waiting for its turn to be presented.
                                  If the queue is full the program must wait(VSYNC, Vertical Sync).  
        VK_PRESENT_MODE_FIFO_RELAXED_KHR: Only differs if the previous one if the app is late and queue was empty at the last vertical blank. Instead of
                                          waiting for next vertical blank image is transferred right away when it arrives. May result in vertical tearing.
        VK_PRESENT_MODE_MAILBOX_KHR: Instead of blocking app when queue is full, the images that are already queued are simply replaced with the newer ones.
                                     Can be used  to render frames as fast as possible while still avoiding tearing, resulting in fewer latency issues than
                                     stanard vsync This is known as triple buffering. 

        VK_PRESENT_MODE_FIFO_KHR is the only guaranteed to be available
    */
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        // We're going to check if VK_PRESENT_MODE_MAILBOX_KHR is available since energy usage is no concern since it allows us to avoid tearing while maintaing low latency
        // Not advised for mobile device
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    /*
        The next and final thing to consider is the swap extent

        The swap extent is the reslution of the swap chain images and it's almost always exactly equal to the reslution of the window we're drawing to in pixels
        Our range of possible resolutions are in our capabilities struct from our swapChainSupportDetails struct
        By defualt vulkan tells us to match the resoluton of the window by setting the width and height in the currentExtent member of our capabilities struct

        Some window managers allow us to differ here and it is indicated by setting the width and height in the currentExtent to a special value:
            max value of uint32_t
        Instead of the default width and height of our window.

        In this case we'll pick the resolution that best matches the window within the minImageExtent and maxImageExtent bounds, but we must specify the resolution in the correct unit.
        
        When creating a window GLFW uses 2 units, width and height.
        Vulkan on the other hand works with pixels so the swap chain extent (resolution) must be specified in pixels not in width and height.
        Depending on your display your screen coords may not correspond to pixels, if your pixel density is higher the resolution of the window in pixels will be larger than the resolution
        screen coordinates.
        So VUlkan Will have to swap the extent for us, we will use glfwGetFrameBufferSize to query the res of the window in pixels before matching it against the min/max image extent.
    */
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        // The reason we check for this is to see if the windowing system (the underlying driver) has a preferred or fixed size for the swap chain images
        // For vulkan our width and height are in pixels so we're checking if the max pixel value is not max val of a 32 bit unsigned int by default
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            // If it is hard coded, that is the ideal situation and we'll just return that as our currentExtent
            // That means our windowing system has provided a concrete size
            return capabilities.currentExtent;
        } else {
            // If our heights and width aren't the max size we will set them ourselves
            // glfw will return our frame buffer width and height to these variables as an int
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            // We'll convert them into unsigned 32 bit ints for vulkan support
            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };
 
            // Clamp the values of width and height between the allowed minimum and maximum capable for my device
            actualExtent.width = std::clamp(actualExtent.width,
                capabilities.minImageExtent.width,
                capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height,
                capabilities.minImageExtent.height,
                capabilities.maxImageExtent.height
            );

            return actualExtent;
        }
    }

    /*
        Function to populate our swapchain struct
        A swap chain is a queue of images ready to be presented to the screen
        Manages the images your application will render into
        Works directly with the present queue family
    */
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
        SwapChainSupportDetails details;

        // Before you create a swap chain you need to know what the surface and physical device are capable of
        // Populate our swap chain support details struct, specifically the capabilities struct
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        // Filling our formats vector
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        // Filling present modes vector
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    /*
        Helper function to load binary data from the files
        will read all of the bytes from a file and return them in a byte array managed by a vector
    */
    static std::vector<char> readFile(const std::string& filename) {
        // We open with 2 flags ate and binary
        // ate: start reading at the end of the file
        // binary: read the files as binary file
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open the file!");
        }

        // The reason we read at the end of the file is we can use the read position to determine the size and allocate a buffer
        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        // Now we can go back to the beginning of the file and read all of the bytes at once
        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    /*
        Before we can pass our code to the graphics pipeline we must first wrap it in a VkShaderModule
        This will take in a buffer with the bytecode and create a shader module out of it

        Shader moduels are just a thin wrapper around the shader bytecode that we've loaded from a file and the functions defined in it
        The compilation and linking of the SPIR-V bytecode to machine code for execution by the gpu doesn't happen until pipeline is created
    */
    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        // the size of the bytecode, inside our code vector, is specified in bytes
        // reinterpret is powerful but potentially dangerous if used incorrectly because it performs a low-level reinterpretation of the bit pattern of a value from one type to another
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }    

    /*
        function that writes the commands we want to execute to a command buffer
        pass in the command buffer as well as the current index of the swap chain image we want to write to
    */
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        // specifies how we're going to use the command buffer, none of which are applicable rn
        beginInfo.flags = 0;
        // only applicable for secondary command buffer infos
        beginInfo.pInheritanceInfo = nullptr;

        // if command buffer was already recorded once, this function will implicitly reset it 
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        // Now to begin drawing we must start a render pass
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        // define the size of the render area, which defines where the shader loads and stores will take place
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;
        //  define the clear values to use for VK_ATTACHMENT_LOAD_OP_CLEAR; defined it to be black and 100% opacity
        // background color of application
        VkClearValue clearColor = {{{ 0.0f, 0.0f, 0.0f, 1.0f}}};
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        // now we can begin our render pass. This is essentially saying that when the command buffer executes start a renderpass with some specific expected settings
        // this is the blueprint for a rendering sequence. since our blueprint expects color attachments this will relate to our draw commands
        // VK_SUBPASS_CONTENTS_INLINE means the render pass commands will be embedded in the primary command buffer and no secondary cmd buffers will be executed
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE); // once again we're not using secondary command buffers

        // now bind the graphics pipeline
        // second option decides if the pipeline object is a graphics or a compute pipeline (we created a graphics pipeline)
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        // now we telling vulkan to bind these vertex buffers to the pipeline essentially
        // since our pipeline is already expecting data from a vertex buffer, when your passing cmds to gpu it'll know to read data from the buffer identified by vertexBuffer
        VkBuffer vertexBuffers[] = { vertexBuffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

        // we did specify a viewport and scissor state for this pipeline to be dynamic so we must set them in the command buffer before issuing our draw command
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        // now we're ready to issue the draw command for the triangle
        // 3rd param is used for instance rendering but we're not doing that so say 1
        vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);

        // now we can end the renderpass
        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to end command buffer!");
        }
    }

    /*
        This is going to be used to combine our memory requirements of the specific buffer we created + our own applications memory to find the right memory type to use
        we do this because our graphics card provides us with different memory types to allocate from that will vary depending on allowed operations and performance characteristics

        sole purpose is to find the correct VkMemoryType from our devices available memory types
        when it's time to actually allocate our vertex information into the GPU we will use the VkMemoryType
    */
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        // so now this part specifies the memory requirements that our gpu provides us so we can find which one will satisfy the requirements our vertex buffer requires
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        /*
            There are two arrays within our physical device memory properties, the heap and the type
            the heap represents the physical pools of memory the gpu has access to as well 

            the different types of memory exist in those heaps
            
            Unlike how primary memory will have a swap space on the secondary memory to load data in and out of primary
            The gpu utilizes a swap space in RAM when VRAM runs out

            Ayways now we can find a memory type that is suitable with our vertex buffer from our gpu
        */
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            /*
                type Filter is used to specify the bit field of memory types that are suitable
                typeFilter will be memoryTypeBits from VkMemoryRequirements in our vertex buffer creation function
                typeFilter is another bitmask for our bitwise operation where each bit represents a flag or a state
                we have this bitmask filled by vkGetBufferMemoryRequirements depending on the memory requirements we stated
                and we just iterate through the device memory properties provided by our gpu moving the 1 over until it's compatible or considered "on" with what the vertex buffer requires + the properties we need
                we filled our VkMemoryRequirements struct using this function vkGetBufferMemoryRequirements so then we can extract the memoryTypeBits and pass it to this

                the memory types array in our memProperties consist of VkMemoryType structs that specify the heap and properties for each type of memory
                it has VkMemoryPropertyFlags propertyFlags; to describe properties of that memory type
                uint32_t heapIndex; and this is the unique identifier  

                so this statement really does 2 things, makes sure the i-th memory type is one that our buffer can actually used (provided by our physical device)         
                and that the i-th memory type also posses all the properties that I need
            */
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    // Helper function to copy from one buffer to another buffer
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
        // luckily we don't have to create a new command pool beacuse VK_QUEUE_TRANSFER_BIT is implicitly support with VK_QUEUE_GRAPHICS_BIT
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        // our command buffer we'll write commands too
        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        // begin info for command buffer so we can start recording
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // only going to use the command buffer once and wait unitl copy operation is finished executing

        // immediately staart recording to command buffer
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin copy command buffer!");
        }

        // this is a struct specifying a buffer copy operation
        VkBufferCopy copyRegion{};

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to end command buffer!");
        }
    }

    /*
        Now that we're going to need multiple buffers 
    */
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkDeviceMemory& bufferMemory, VkBuffer& buffer) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, props);
        allocInfo.allocationSize = memRequirements.size;

        // this is the exact moment we allocate memory onto the gpu
        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            std::runtime_error("failed to allocate memory on gpu!");
        }

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    // Store and initiate each vulkan object
    void initVulkan() {
        // Is called on it's own to initialize vulkan
        createInstance();
        setUpDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFrameBuffers();
        createCommandPool();
        createVertexBuffer();
        createCommandBuffers();
        createSyncObjects();
    }

    /*
        Creating a vertex buffer doesn't have it's own vk command so we'll create a general buffer

        the vertex buffer represents the GPUs named access point to that data that you stored with in it
    */
    void createVertexBuffer() {
        VkDeviceSize size = sizeof(vertices[0]) * vertices.size();
        // this buffer is going to have a transfer usage, we'll transfer it into the vertex buffer once it's on the gpu and written to
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferMemory, stagingBuffer);
        /*
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO
            // we do this so we can get the size of the actual data we own
            // doing just sizeof(vertices) would be the size of std::vector<Vertex> which is what our vertices variable is
            bufferInfo.size = sizeof(vertices[0]) * vertices.size();
            // What kind of buffer is it
            bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            // buffers can be owned by a specific queue family or be shared between multiple at the same time
            // our vertices will only be used by the graphics queue so we'll stick to this
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            // At this point our buffer is created but it's not actually filled with any memory
            if (vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer) != VK_SUCCESS) {
                std::runtime_error("failed to create vertex buffer!");
            }

            // while we've created the buffer itself we haven't allocated any memory into it
            // the first step of allocating memory for the buffer is to query the memory requirements
            // this tells us what kind of memory the buffer needs
            
            VkMemoryRequirements memRequirements;
            vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements); // fills our struct for us

            // now it's time to allocate the vertex information
            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memRequirements.size;
            // we check if a memory type of a specific type specified by our vertex buffer memory requirements func is available by our gpu
            // we then check if that memory type has either of these flags available to them so we can write to that specific heap from our cpu
            allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            // now we will allocate memory to a specific heap on the gpu, request and reserve a specific block of physical memory on the gpu
            if (vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory) != VK_SUCCESS) {
                std::runtime_error("failed to allocate vertex buffer memory!");
            }

            // since memory allocation didn't return an error we can now associate this memory with the buffer
            vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);
        */

        // now we need to fill our vertex buffer, with a data variable that points to a memory location on our gpu
        // even though we linked our vertex buffer to a specific chunk of memory on our gpu, we still can't directly write to it yet
        // represents a host accessible virtual address to memory on our gpu
        void* data;
        // this allows us to access a region of the specified memory resource defined by an offset and size
        // this creates the bridge between the address spaces of the vetex data and the space on our gpu
        // it makes a region of the GPU-allocated memory directly accessible to your CPU code
        vkMapMemory(device, stagingBufferMemory, 0, size, 0, &data);
        // now we can copy some data into that mapped memory
        // just transfers the vertices data into the VkDeviceMemory
        memcpy(data, vertices.data(), (size_t) size);
        // then unmap that memory
        vkUnmapMemory(device, stagingBufferMemory);

        /*
            So from the above creation of the staging buffer, we created a staging buffer with a source transfer usage bit flag
            we find the memory requirements the buffer requires and allocate the memory onto our gpu using our specified usage bit flag and property bit flag
            then we bind the buffer to that region of memory to be used by our gpu later on
        */

        // represents our actual vertex buffer that we're going to use VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT because that is the most optimal memory flag bit for the gpu to read from
        // we're also going to use it as a destination buffer during a transfer
        // now our vertex buffer has a memory type that is device local
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBufferMemory, vertexBuffer);
    }

    /*
        This is how we're going to create our semaphores and fence
    */
    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                    throw std::runtime_error("failed to create semaphores!");
            }
        }
    }

    /*
        Commands in vulkan (like draw operations and memory transfers) are not executed directly using function calls
        you have to record all operations you want to perform in command buffer objects
        this allows you to submit all commands together and vulkan can efficiently process the commands since they're all together
        this also allows command recording in multiple threads if desired

        command buffers are executed by submitting them on one of the device queues (which is managed by the command pool)
    */
    void createCommandBuffers() {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        // specifies if the allocated command buffers are primary or secondary command buffers
        // primary can be submitted to a queue for execution but cannot be called from other command buffers
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        // only creating one command buffer
        allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command buffers!");
        }
    }

    /*
        we need to create a command buffer before we must create a command pool that manages the memory that is used to allocate command buffers and is attached to a specific queue family
        each command pool can only allocate command buffers that are submitted on a single queue type
    */
    void createCommandPool() {
        // The queue families we'll be sending rendering operatiosn to
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        // simple struct VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT allows command buffers to be rerecorded individually or else they'll be reset together
        // we'll be recording a command buffer every frame so we want to be able to reset and rerecord over it
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        // takes a command pool create info and a reference to the actual command pool
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    // Now we can create our framebuffers
    // Framebuffer just represents a single instance of a rendering target, in our case it'll be an instance of an image and some of it's information
    // Think of the renderpass as a recipe and the framebuffer as the tools needed to create our recipe (which is our iamge in this case)
    void createFrameBuffers() {
        // first we must resize our framebuffer vector by the size of our vkImageViews
        swapChainFramebuffers.resize(swapChainImageViews.size());

        for (int i = 0; i < swapChainImageViews.size(); i++) {
            // the attachment for each swap chain is going to be an image view
            VkImageView attachments[] = {
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    /*
        before creating the pipeline we must tell vulkan about the framebuffer attachments that will be used while rendering
        specify how many color and depth buffers there will be, how many samples to use for each, and how their contents should be handled throughout rendering operations
        describe format and the way we load and store an image
        where the GPU will write color, depth, and stencil information generation by rendering commands

        the renderpass is a sequence of rendering operations including the framebuffer attachments that will be used (color, depth, stencil)
        how they'll be loaded and stored, and any synchronization dependencies between subpasses

        THINK OF IT AS A BLUEPRINT FOR THE VULKAN OF A SEQUENCE OF OPERATIONS AND HOW THEY'LL INTERACT WITH AN IMAGE
    */
    void createRenderPass() {
        // in our case we'll have just a single color buffer attachment represented by one of the images from the swap chain
        // color attachments are iamges that will hold color data generated by the fragment shader
        VkAttachmentDescription colorAttachment{};
        // format should match format of our swap chain images
        colorAttachment.format = swapChainImageFormat;
        // We're not doing anything with multisampling so we're only doing 1 sample
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        // Applies to color and depth data
        // we use clear for load op so we can clear the framebuffer to black before drawing a new frame
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        // rendered contents will be stored in memory and can be read later
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        // applies to stencil data, we're not going to use the stencil buffer so the results of loading and storing don't matter
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        // Textures and framebuffers in vulkan are represented by VkImages in vulkan with a certain pixel format
        // however the layout of the pixels in memory can change based on what you're trying to do, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR represent images to be presented in the swap chain
        // what's important to know is that images are going to be transitioned to a specific layout that are suitable for the operations they're going to be involved in
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // layout before rendering pass begins
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // layout to automatically transition to after
        /*
            A single render pass can consist of multiple sub passes
            subpasses are a subsequent rendering operations that depend on the contents of framebuffers from previous renderpasses
            grouping them together in one render pass allows vulkan to reorder them in a way that can conserve memory bandwidth for potentially better performance

            every subpass references one or more of the attachments that we've described using the structure in the previous sections
        */
        VkAttachmentReference colorAttachmentRef{};
        // specifies which attachment to reference by its index in the attachment descriptions array
        // our array only has a single VkAttachmentDescription which is the colorAttachment
        colorAttachmentRef.attachment = 0;
        // specifies which layout we want the attachment to have during a subpass that uses this reference
        // we intend to use the attachment to function as a color buffer and this layout will give us the best performance, as its name implies
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        /*
            Now for the creation of the subpass itself, this is made clear to be a graphics subpass
            automatically take care of image layout transitions (the layout that within our shaders)
            at least 1 subpass in every renderpass
        */
        VkSubpassDescription subpass{};
        // pipeline type that is supported, we created a graphics pipeline
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        // Now onto creating our actual render pass
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        /*
            The layout transitions are controlled by subpass dependencies which specify memory and execution dependencies between subpasses
            we only have one subpass right now but the operations right before and right after this subpass also count as implicit subpasses

            The two built in dependencies that take care of this layout transition at the start of the renderpass and at the end of the renderpass
            the one at the beginning does not occur at the right time, it assumes the transition occurs at the start of the pipeline but we haven't acquired the image yet at that point

            there are 2 ways to handle it, either change the wait stage in the submit info function or we can make the render pass wait for the VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT stage
            the 2nd option is what we'll do here
        */
        VkSubpassDependency dependency{};
        /*
            specify the indices of the dependency and the dependent subpass
            VK_SUBPASS_EXTERNAL refers to the implicit subpass before or after the render pass depending on whether it is specified in srcSubpass or dstSubpass
            src is who I'm dependent, almost always have a lower index than dst
            
            this really defines the external renderpass stage before the subpass, we can also use this for other renderpasses
        */
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        // this must always be higher than our src to prevent ycles in the dependency graph
        // who I am, current index of our subpass which is 0 the first and only one
        dependency.dstSubpass = 0;
        // specifies which operations to wait on and the stage these operations occur at
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        // the operation that should wait on this are in the color attachment stage and involve the writing of the color attachment
        dependency.srcAccessMask = 0;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    /*
        Prepare for a fat paragraph:
        The graphics pipeline is just a sequence of operations your gpu performs
        These operations take the vertices textures of your meshes, the pixels in the render targets, etc etc

        simplified overview:
            (fixed-function; tweak with params but are predefined) Vertex/Index buffer: buffer that holds raw vertex data
            (programmable; you can upload your own code to gpu to apply operations) Input assembler: collects the raw vertex data from the buffers you specify (May use the index buffer to reuse vertex data without having to duplicate)
            (programmable) Vertex Shader: ran for every vertex and generally applies transformations to turn vertex positions from the model space to the screen space 
            (programmable) Tessellation: allows you to subdivide geometry based on certain rules to increase mesh quality. used for surfaces like brick walls and stairs to prevent it from looking flat up close
            (programmable) Geometry shader: ran on every primitive (line, point, triangle) and can discard it or output more primitives than came in. Similar to tesselation but more flexible.
            (fixed-function) Rasterization: discretizes the primitives into fragments (pixel elements that they fill on the framebuffer). 
            (programmable) Fragment shader: invoked for every fragment that survives and determines which framebuffer the fragments are written to and with which color and depth values
            (fixed-function) Color blending: applies operations to mix different fragments that map to the same pixe in the framebuffer

        Graphics pipeline in vulkan is immutable
        If you wish to make a change, most of the times, you'd have to completely recreate the graphics pipeline

        unlike earlier APIs vulkan shader code has to be specified in a bytecode format opposed to a human read-able syntax GLSL and HLSL
        The advantages of using this format is that the compilers written by GPU vendors to turn shader code into native code are significantly less complex
        Thankfully Khronos has released their own vendor-independent compiler that compiles GLSL to SPIR-V.
        We'll be using glslc.exe created by google. Which is the compiler for compiling glsl to spir-v which is already included in the vulkan sdk

        GLSL is a shading language with a c-style syntax. Programs written in it have a main function that is invoked for every object.
        GLSL uses global variables for input and output over using parameters and return.
    */
    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("../shaders/vert.spv");
        auto fragShaderCode = readFile("../shaders/frag.spv");

        auto vertShaderModule = createShaderModule(vertShaderCode);
        auto fragShaderModule = createShaderModule(fragShaderCode);

        // Vert Shader
        // To actually use the shaders we'll need to assign them to a specific pipeline stage through VkPipelineShaderStageCreateInfo
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        // Set our vertShader into the vertex shader stage
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        // specify shader module with the code and the function to invoke (known as the entry point)
        // it's possible to combine multiple fragment shaders into a single shader module and use different entry points to differentiate between their behaviors
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        // Frag Shader
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        // Array that holds both of these structs
        // This finalizes everything we need to do for our programmable stages, now we need to do the fixed-function
        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        /*
            Even the majority of the pipeline state (which is just the overall configurations and properties that make up the entire pipeline) is immutable (unchanging)
            There are certain aspects that can be changed without having to recreate the entire pipeline

            Anything inside this vector will be ignored during configuration
            This causes the configuration of these values to be ignored and you'll be able to specify the data at draw time
            results in a much more flexible setup
        */
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        /*
        1. Vertex Input
            describes the format of the vertex data that will be passed to the vertex shader; describes them in one of 2 ways
            bindings: spacing between data and whether the data is per-vertex or per-instance
            attribute descriptions: type of the attributes passed to the vertex shader, which binding to load them from and at which offset

            Since we hardcoded this vertex data into our shader for now, we'll specify that there is no vertex data to load for now and get back to it later when we create a vertex buffer
        */
        auto bindingDescriptions = Vertex::bindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescriptions;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        /*
        2. Input Assembly
            describes what kind of geometry will be drawn from the vertices and if primitive restart should be enabled
        */
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        /*
        3. Viewport and scissors (dynamic states)
            The view port describes the region of the framebuffer that the ouput will be rendered too
            Almost always (0, 0) to the width and height

            The size of the swap chain and its images may differ from the width and height of window
            the swap chain images will be used as framebuffers later on so we should stick to their size
            viewport define the transformation from the image to the framebuffer
        */
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapChainExtent.width;
        viewport.height = (float) swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        // Scissor rectangles define in which regions pixels will actually be stored
        // Any pixels outside the scissor rectangles will be discarded by the rasterizer
        // It's more like a filter than a transformation
        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;

        

        // then you only need to specify their count at pipeline creation time
        // The actual viewport and scissor rectangle will then later be set up at drawing time
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        /*
        4.  Rasterizer
            Takes the geometry that is shaped by the vertices from the vertex shader and turns it into fragments to be colored by the fragment shader
            It also performs depth testing, face culling, and the scissor test
            it can also be configured to output fragments that fill entire polygons or just the edges (wireframe rendering)

            if depth clamp was true the geometry would never pass through the rasterizer stage
            Disables any output to the framebuffer
        */
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        // Polygon mode determines how fragments are generated for geometry
        // There is Fill(fill with fragments), Line(edges are drawn as lines), and point(polygon vertices are drawn as points)
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        // describes the thickness of lines in terms of number of fragments. Anything higher than 1.0f requires you to enable wideLines gpu feature
        rasterizer.lineWidth = 1.0f;
        /*
            Determines the type of face culling to use, 
            You can disable it, cull the front, cull the back, or both
            Checks all the faces of a shape facing the viewer and renders those while discarding all that are back facing
            You can do the same by just rendering the faces in the back and not render the ones facing the viewer

            frontface variables specifies the vertex order for faces to be considered front-facing 
        */
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        // The rasterizer can alter the depth values by adding a constant value or biasing them based on the fragments slope
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f;
        rasterizer.depthBiasClamp = 0.0f;
        rasterizer.depthBiasSlopeFactor = 0.0f;

        /*
            This struct configures multisampling which is one of the ways to perform anti-aliasing
            It combines the fragment shader results of multiple polygons that rasterize to the same pixel
            Most noticeable around edges

            disabling it for now
        */
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f; // optional
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;

        /*
            After a frag shader has returned a color, it must be combined with a color that's already in the framebuffer
            This is known as color blending and there's 2 ways to do this
            1. Mix the old and new value to produce final color
            2. Combined the old and new value using a bitwise operation

            1. this per-framebuffer struct allows you to configure the first way of color blending
        */
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        // 2. This structure references the array of structs for all of the framebuffers and allows you to set blend constants that you can use as blend factors in the calculations
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        // 4 available blend constants in the struct
        colorBlending.blendConstants[0] = 0.0f;
        colorBlending.blendConstants[1] = 0.0f;
        colorBlending.blendConstants[2] = 0.0f;
        colorBlending.blendConstants[3] = 0.0f;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 0;
        pipelineLayoutInfo.pSetLayouts = nullptr;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        // now we can create our actual pipeline
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        // we start by referencing the array of our pipeline shader stage
        // Our vertex and fragment shaders
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        // Then we reference all of the structs describing the fixed-function stage
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        // Then we get our pipeline layout
        pipelineInfo.layout = pipelineLayout;
        // then we can finally reference our renderpass and the index of our sub pass
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        // these are optional, vulkan lets you create a new pipeline by deriving from an existing pipeline
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = -1;

        // the vkCreateGraphicsPipelines function is designed to create multiple pipelines using multiple vkGraphicsPipelineCreateInfo objects
        // the second param references an optional VkPipelineCache object which can be used to store and reuse data relevant to pipelien creation across multiple calls
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        // since the compilation and linking of the SPIR-V doesn't happen until graphics pipeline is created we can delete at the end of function
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
    }

    void createImageViews() {
        // We'll resize our image views vector to be the same size the total of our swap chain images
        swapChainImageViews.resize(swapChainImages.size());

        // Iterate through over all of the swap chain images
        // We're creating an image view for each swap chain image
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = swapChainImages[i];

            // The view type and format specify how image data should be interpreted
            // view type param allows you to treat images as 1D textures, 2D textures, 3D textures
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            // format, once again, is the surface format and that means how the pixels in an data is stored in memory like r8g8b8a8
            createInfo.format = swapChainImageFormat;

            /*
                The components field allows you to swizzle the color channels around
                You can map all of the channels to the red channel for monochrome for example
                we'll stick with default
            */
           createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
           createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
           createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
           createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

           // The subresourceRange field describes what the image's purpose is and which part of the image should be accessed
           // Our images will be used as color targets without any mipmapping levels or multiple layers
           createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
           createInfo.subresourceRange.baseMipLevel = 0;
           createInfo.subresourceRange.levelCount = 1;
           createInfo.subresourceRange.baseArrayLayer = 0;
           createInfo.subresourceRange.layerCount = 1;

           // Now we can actually create the image views
           if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
           }
        }
    }

    void cleanupSwapChain() {
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }

        // since we explicitly created the image views ourselves we need to loop through it to delete everything
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    // there are some circumstances that aren't being handled correct such as the swap chain not being compatible with a window surface because the window size changes
    // we have to catch these certain events
    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }
        // we call this function so we don't touch resources that may still be in use
        vkDeviceWaitIdle(device);

        // make sure the old versions of these objets are cleaned up before recreation
        cleanupSwapChain();

        // swap chain will have to be recreated because that is our queue of images
        createSwapChain();
        // image views will have to be recreated because they are based on our swap chain images
        createImageViews();
        // frame buffers directly depend on the swapchain 
        // we don't recreate the framebuffer for simplicity but we don't since there isn't a chance of the image format to change
        createFrameBuffers();
    }

    // Now we can finally create our swap chain
    void createSwapChain() {
        // Just finds all of our available and suitable swap chain surface formats, capabilities, and present modes
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D swapExtent = chooseSwapExtent(swapChainSupport.capabilities);

        // Specifies the minimum number of images we want in the swap chain
        // We add 1 so we can ensure that we don't have to wait on the driver to complete internal operations before we can acquire another image
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        // We need to also ensure to not exceed the maximum number of images
        // 0 means there is no maximum so we check to make sure there is a maximum  and image count is greater than that maximum
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        // Now we can fill the struct to create our swapchain
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        // After we specify the surface the swap chain is tied to the details of the swap chain images are specified
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = swapExtent;
        // Specifies amount of layers each image consists of
        // Should always be 1 unless you develop a stereoscopid 3d app
        createInfo.imageArrayLayers = 1;
        // Specifies what kind of operations we'll use the images in the swap chain for
        // Since we're rendering directly to them in this tutorial, we'll be usinng them as a color attachment
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // Could use VK_IMAGE_USAGE_TRANSFER_DST_BIT instead if you wanted it rendered to a separate file for post processing

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        // if when searching for our graphics family and present family, separate values are returned
        // if they are separate queues we'll be drawing on the images in the swapchain from the graphics queue and then submitting to them from the presentation queue
        if (indices.graphicsFamily != indices.presentFamily) {
            // Images can be used across multiple queue families without explicit ownership transfers
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            // Concurrent mode requires you to specify in advanced that the ownership of an image will be shared
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            // An image is owned by one queue family at a time and ownership must be explicitly transferred before using it in another queue family 
            // best performance
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0; // optional
            createInfo.pQueueFamilyIndices = nullptr; // optional
        }

        // We can specify that certain tranform should be applied to images in the swap chain if it is supported, like a 90 degree clockwise rotation or horizontal flip
        // Specifying the current transform means you do not want any tranformation
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        // The compositeAlpha specifies the alpha channel should be used for blending with other windows in the windowing system
        // You almost always want to ignore the alpha channel
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        // if the clipped member is set to true that means we don't care about the color of pixels that are obscured
        createInfo.clipped = VK_TRUE;
        // It's possible your swap chain becomes invalid or unoptimized while your app is running, like if the window is resized
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        // Create swap chain store it in swap chain
        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = swapExtent;
    }


    /*
        Keep in mind you must create a swapchain if you want to present an image to the surface using the presentQueue if available
        Represents the connection between your vulkan instance and a specific output desitnation provided by a windowing system
        Since VUlkan is platform agnostic and every os has a different way to handle windows you need a way to interface with these different systems
        To do this you must write vulkan operations to a surface that then links to a window that it'll present to

        Is an object that offers an abstraction to your devices windowing system
    */
    void createSurface() {
        // Glfw is creating a vulkan window surface linked to our glfw window
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    // After we select our physical device, we need to set up a logical device to interface with it
    void createLogicalDevice() {
        // Describes the features we want to use as well as the queues to create now that we've queried which ones are available
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        float queuePriority = 1.0f;
        for (auto queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // Some device features that we'll be using later on
        // Won't do anything so it'll default to false but we need this to create logical device
        VkPhysicalDeviceFeatures deviceFeatures{};

        // Actual info to create logical device
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        // Set the pointers to our queue create info and device features
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        createInfo.pEnabledFeatures = &deviceFeatures;

        // Similar to vulkan instance but this time we're creating it to be device specific
        // Enabling our device extensions (VK_KHR_swapchain)
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        // Done on a device level to perform checks on the specific operations performed on device
        if (enableValidationLayers) {
            // Using the same VK_LAYER_KHRONOS_validation to ensure validation of rendering commands and resource usage
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

    }

    // After intializing the vulkan library through a vkInstance we need to look for and select a graphics card
    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // This function right now only picks the very first physical device (integrated or dedicated)
        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("Failed to find a suitable GPU!");
        }
    }

    // Our debug messenger essentially acts as the central hub for all of our
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

    /*
        The general pattern that object creation function params in vulkan follow is:
        Pointer to struct with creation info
        Pointer to custom allocator callbacks
        Pointer to the varaible that stores the handle to the new object
    */
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

        /*
            Now specified everything Vulkan needs to create an instance and we can finally issue the vkCreateInstance call
            Creating a VKInstance object initializes the vulkan library and allows the app to pass info about itself to the implementation
            Check to see if our Vulkan instance was successfully created
        */
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create an instance!");
        }
    }

    
    /*
        At a high level, rendering a frame in vulkan has several common steps:
            1. Wait for previous frame to finish
            2. acquire an image from the swap chain
            3. record a command buffer which draws the scene onto that image
            4. submit the recorded command buffer
            5. present the swap chain image

        by design there is a slight error, the fence starts in the unsignaled state at the first frame waiting for a previous frame (that doesn't exist) to finish executing
        this causes our program to be stuck
        We set our semaphore to the signaled state in the first frame to work around this
    */
    void drawFrame() {
        // At the start of the frame we wait until the previous frame has finished so command buffer and semaphores are available to use
        // timeout disabled
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        // now we will acquire an image from our swap chain to render to
        uint32_t imageIndex;

        /*
            luckily vulkan will automatically tell us if our swap chain becomes incompatible or changes
            as soon as one of our swap chain images aren't compatible with the swap chain we'll receive an error message from our validation layer
            when that occurs we will reset it depending on the current settings of our surface
            VK_ERROR_OUT_OF_DATE_KHR means the swap chain has become incompatible with the surface and can no longer be used for rendering (usually screen size changed)
            VK_SUBOPTIMAL_KHR means that the swap chain can still be used to present to the surface but the surface properties aren't matched exactly
        */
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        /*
            after waiting we manually reset the fence to the unsignaled state because we signaled it in the beginning to continue
            We want to avoid resetting our fence until we know we will for sure be submitting work with it
            If we reset the fence and our program returns an error that caused our swap chain to be reset below, it will return to drawframe as unsignaled
            This will cause our program to half forever since our fence will remain in the unsignaled state preventing our application from ever getting a signal to continue
            only reset if we an image is acquired for submission
        */
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        // reset command buffer so we can use for recording to
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        // now we call the function to record the commands we want
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        // queue submission and synchronization will be configured through this struct
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        // this is the semaphore used specifically between operations
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        // specifies which semaphore to signal once the command buffer has finished execution
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
        // specifies which semaphores we'll be using and which stages of the pipeline to wait; we're waiting on the color attachment stage
        // the wait will occur at the color attachment stage, aka our render targets aka our image views aka our images
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        // which command buffers to actually submit for execution (we only have the one)
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        // now we submit the command buffer to the graphics queue using this function
        // now the next frame the cpu will wait for this command buffer to finish executing before it records new commands in it
        // This is the hand off from the CPU to the GPU
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        /*
            Finally at the final stage which is submitting the result back to the swap chain for presentation eventually
        */
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        // specifies which semaphores to wait on before presentaiton can happen
        // since we wanna wait for the command buffer to finish execution (drawing our triangle) we take the semaphores which will be signalled and wait on them
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        // specify the swap chains to present images to and the index of the image for each swapchain
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        // optional param that allows you to specify an array of VkResult values to check for every individual swap chain if presentation was successful
        // not necessary since we're only using a single swap chain
        presentInfo.pResults = nullptr;

        // submits the request to present an image to the swap chain
        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image");
        }

        // so we can only get 0 and 1 value when indexing for our next frame
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // Loop that iterates until the window is closed in a moment
    void mainLoop() {
        // Keep the application running until an error occurs or the window is closed
        // While window is open
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents(); // Checks for events 
            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }

    // Once window is is closed we'll deallocate used resources
    // Every Vulkan object that we create needs to be destroyed when we no longer need it
    // It is possible to perform automatic resource management using RAII or smart pointers
    void cleanup() {
        cleanupSwapChain();

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);

        for (auto semaphore : imageAvailableSemaphores) {
            vkDestroySemaphore(device, semaphore, nullptr);
        }

        for (auto semaphore : renderFinishedSemaphores) {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
        
        for (auto fence : inFlightFences) {
            vkDestroyFence(device, fence, nullptr);
        }

        vkDestroyCommandPool(device, commandPool, nullptr);

        vkDestroyPipeline(device, graphicsPipeline, nullptr);

        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);

        vkDestroyRenderPass(device, renderPass, nullptr);

        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        // Destroy our created surface
        vkDestroySurfaceKHR(instance, surface, nullptr);
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