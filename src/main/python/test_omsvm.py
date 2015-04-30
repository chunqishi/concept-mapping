import unittest
import codecs
import os
from omsvm import line_process, file_process, uniq_arrs

class Test(unittest.TestCase):
    def setUp(self):
        s ="""
        .{http://www.omg.org/XMI}XMI.#text
        .{http://www.omg.org/XMI}XMI.@{http://www.omg.org/XMI}version	2.0
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[0].#tail
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[0].@begin	2417
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[0].@componentId	GATE Sentence Splitter
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[0].@end	2464
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[0].@sofa	1
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[0].@{http://www.omg.org/XMI}id	1600
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[10].#tail
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[10].@begin	447
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[10].@componentId	GATE Sentence Splitter
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[10].@end	571
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[10].@sofa	1
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[10].@{http://www.omg.org/XMI}id	1581
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[11].#tail
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[11].@begin	152
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[11].@componentId	GATE Sentence Splitter
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[11].@end	239
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[11].@sofa	1
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[11].@{http://www.omg.org/XMI}id	1578
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[12].#tail
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[12].@begin	240
        .{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Sentence[12].@componentId	GATE Sentence Splitter
        """
        self.fl = "test_file_process.tmp"
        outputstream = codecs.open(self.fl, mode='w', encoding='utf-8')
        outputstream.write(s)
        outputstream.close()

    def tearDown(self):
        os.remove(self.fl)

    def test_line_process(self):
        self.assertEqual(line_process("{http://www.omg.org/XMI}XMI.{http:///org/apache/uima/examples/opennlp.ecore}Token[108].@posTag	JJ"), ("JJ", "XMI.Token.@posTag"))

    def test_file_process(self):
        data, target  = file_process(self.fl)
        # print str(res)
        self.assertEqual(len(data), 23)

    def test_uniq_arrs(self):
        data, target = file_process(self.fl, [], [], {})
        print len(data), str(data)
        data, target = uniq_arrs(data, target)
        print len(data), str(data)
        self.assertTrue(len(data)< 23)

if __name__ == '__main__':
    unittest.main()