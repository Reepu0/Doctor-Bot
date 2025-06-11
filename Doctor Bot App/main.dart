import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.light(useMaterial3: true),
      home: const HomeScreen(),
    );
  }
}

//HomeScreen
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  void _showInfoDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(20)),
          title: const Text(
            "About Doctor Bot",
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 20),
          ),
          content: const Text(
            "Doctor Bot is a virtual assistant designed to help users get quick answers to health-related queries. It uses advanced AI to analyze symptoms and provide guidance. This app is not a substitute for professional medical advice, diagnosis, or treatment.",
            style: TextStyle(fontSize: 16),
          ),
          actions: [
            TextButton(
              child: const Text(
                "OK",
                style: TextStyle(fontSize: 16, color: Colors.blueAccent),
              ),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Doctor Bot',
          style: TextStyle(
            color: Colors.white,
            fontSize: 22,
            fontWeight: FontWeight.bold,
          ),
        ),
        centerTitle: true,
        backgroundColor: const Color(0xFF1D3557),
        actions: [
          IconButton(
            icon: const Icon(Icons.help_outline, color: Colors.white),
            onPressed: () => _showInfoDialog(context),
          ),
        ],
      ),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [
              Color(0xFF457B9D),
              Color(0xFFA8DADC),
            ],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircleAvatar(
                backgroundColor: Colors.white,
                radius: 80,
                child: Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Image.asset('assets/images/health.png'),
                ),
              ),
              const SizedBox(height: 30),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 40, vertical: 15),
                  backgroundColor: const Color(0xFF1D3557),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(25),
                  ),
                ),
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => const DoctorChat()),
                  );
                },
                child: const Text(
                  'Start Chat',
                  style: TextStyle(
                    fontSize: 18,
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}


//DoctorChat
class DoctorChat extends StatefulWidget {
  const DoctorChat({super.key});

  @override
  _DoctorChatState createState() => _DoctorChatState();
}

class _DoctorChatState extends State<DoctorChat> {
  final List<Map<String, dynamic>> _messages = [];
  final TextEditingController _controller = TextEditingController();
  final ImagePicker _picker = ImagePicker();
  File? _selectedImage;
  final ScrollController _scrollController = ScrollController();
  @override
void initState() {
  super.initState();
  // إضافة رسالة الترحيب عند فتح الدردشة
  Future.delayed(Duration.zero, () {
    setState(() {
      _messages.add({
        'message': 'Welcome to Doctor Bot! Enter a description of the disease or a picture of the disease.',
        'isUser': false,
      });
    });
  });
}

  void _sendMessage(String text) async {
    if (text.trim().isNotEmpty) {
      setState(() {
        _messages.add({'message': text, 'isUser': true});
        _controller.clear();
        _scrollToBottom();
      });

      try {
        final response = await http.post(
        //Uri.parse('http://10.0.2.2:5000/api/chat'),
        Uri.parse('http://192.168.1.5:5000/api/chat'),
          headers: <String, String>{
            'Content-Type': 'application/json; charset=UTF-8',
          },
          body: jsonEncode(<String, String>{
            'message': text,
          }),
        );

        if (response.statusCode == 200) {
          final responseData = jsonDecode(response.body);
          setState(() {
            _messages.add({'message': responseData['response'], 'isUser': false});
            _scrollToBottom();
          });
        } else {
          setState(() {
            _messages.add(
                {'message': 'Error: Unable to get response.', 'isUser': false});
            _scrollToBottom();
          });
        }
      } catch (error) {
        setState(() {
          _messages.add({
            'message': 'Error: Could not connect to the server.',
            'isUser': false
          });
          _scrollToBottom();
        });
      }
    }
  }

  Future<void> _uploadImage(File image) async {
    try {
      //final request = http.MultipartRequest('POST', Uri.parse('http://10.0.2.2:5000/api/image'));
      final request = http.MultipartRequest('POST', Uri.parse('http://192.168.1.5:5000/api/image'));
      request.files.add(await http.MultipartFile.fromPath('image', image.path));
      final response = await request.send(); 

      if (response.statusCode == 200) {
        final responseData = await response.stream.bytesToString();
        setState(() {
          _messages.add({'message': jsonDecode(responseData)['response'], 'isUser': false});
          _scrollToBottom();
        });
      } else {
        setState(() {
          _messages.add({'message': 'Error: Unable to get response.', 'isUser': false});
          _scrollToBottom();
        });
      }
    } catch (error) {
      setState(() {
        _messages.add({'message': 'Error: Could not connect to the server.', 'isUser': false});
        _scrollToBottom();
      });
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
        _messages.add({
          'message': 'Image selected',
          'isUser': true,
          'image': _selectedImage
        });
        _scrollToBottom();
      });
      await _uploadImage(_selectedImage!);
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Chat with Doctor Bot',
          style: TextStyle(
            color: Colors.white,
            fontSize: 22,
            fontWeight: FontWeight.bold,
          ),
        ),
        backgroundColor: const Color(0xFF1D3557),
        centerTitle: true,
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final message = _messages[index];
                return Align(
                  alignment: message['isUser']
                      ? Alignment.centerRight
                      : Alignment.centerLeft,
                  child: Padding(
                    padding: const EdgeInsets.symmetric(
                        vertical: 8, horizontal: 16),
                    child: message['image'] != null
                        ? ClipRRect(
                            borderRadius: BorderRadius.circular(15),
                            child: Image.file(
                              message['image'],
                              width: 200,
                              height: 200,
                              fit: BoxFit.cover,
                            ),
                          )
                        : Container(
                            padding: const EdgeInsets.all(12),
                            constraints:
                                const BoxConstraints(maxWidth: 250),
                            decoration: BoxDecoration(
                              color: message['isUser']
                                  ? const Color(0xFF1D3557)
                                  : const Color(0xFF457B9D),
                              borderRadius: BorderRadius.circular(15),
                            ),
                            child: Text(
                              message['message'],
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                              ),
                            ),
                          ),
                  ),
                );
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(10),
            child: Row(
              children: [
                IconButton(
                  icon: const Icon(Icons.camera_alt, color: Color(0xFF1D3557)),
                  onPressed: () => _pickImage(ImageSource.camera),
                ),
                IconButton(
                  icon: const Icon(Icons.photo, color: Color(0xFF1D3557)),
                  onPressed: () => _pickImage(ImageSource.gallery),
                ),
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: InputDecoration(
                      hintText: 'Type your message...',
                      filled: true,
                      fillColor: Colors.white,
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(30),
                        borderSide: BorderSide.none,
                      ),
                      contentPadding: const EdgeInsets.symmetric(
                          vertical: 10, horizontal: 15),
                    ),
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.send, color: Color(0xFF1D3557)),
                  onPressed: () {
                    _sendMessage(_controller.text);
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
