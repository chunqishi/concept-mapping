package paper;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.util.CoreMap;
import gate.*;
import gate.Annotation;
import gate.util.Out;
import gate.util.persistence.PersistenceManager;
import opennlp.uima.namefind.NameFinder;
import opennlp.uima.postag.POSModelResourceImpl;
import opennlp.uima.postag.POSTagger;
import opennlp.uima.sentdetect.SentenceDetector;
import opennlp.uima.sentdetect.SentenceModelResourceImpl;
import opennlp.uima.tokenize.Tokenizer;
import opennlp.uima.tokenize.TokenizerModelResourceImpl;
import opennlp.uima.util.UimaUtil;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.cas.Feature;
import org.apache.uima.cas.Type;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.CasToInlineXml;
import org.apache.commons.io.FileUtils;
import org.json.JSONObject;
import org.json.XML;
import uima.CasWriter;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.net.URL;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
import static org.apache.uima.fit.factory.ExternalResourceFactory.createDependencyAndBind;
import static org.apache.uima.fit.factory.JCasFactory.createJCasFromPath;

/**
 * Created by lapps on 3/27/2015.
 *  1. compare opennlp uima, stanford, gate opennlp.
 *  2.
 */
public class Compare {

    public static String gate(String path) throws Exception {
        Out.prln("Initialising GATE...");
        File gatedir = FileUtils.toFile(Compare.class.getResource("/gate/"));
        System.setProperty("gate.site.config", new File(gatedir,"gate.xml").getPath());
        System.setProperty("gate.plugins.home", new File(gatedir, "plugins").getPath());
        Gate.init();
        Out.prln("...GATE initialised");
        // initialise ANNIE (this may take several minutes)
        Out.prln("Initialising ANNIE...");
        // load the ANNIE application from the saved state in plugins/ANNIE
        File pluginsHome = Gate.getPluginsHome();
//        File pluginsHome = new File("C:\\Program Files\\GATE_Developer_8.0\\plugins");
        File anniePlugin = new File(pluginsHome, "ANNIE");
        File annieGapp = new File(anniePlugin, "ANNIE_with_defaults.gapp");
        CorpusController annieController =
                (CorpusController) PersistenceManager.loadObjectFromFile(annieGapp);
        Out.prln("...ANNIE loaded");
        // create a GATE corpus and add a document for each command-line
        // argument
        Corpus corpus = Factory.newCorpus("StandAloneAnnie corpus");
        String [] files  = new String []{new File(path).toURI().toURL().toExternalForm()};
        Out.prln("files.length = " + files.length);
        for(int i = 0; i < files.length; i++) {
            URL u = new URL(files[i]);
            FeatureMap params = Factory.newFeatureMap();
            params.put("sourceUrl", u);
            params.put("preserveOriginalContent", new Boolean(true));
            params.put("collectRepositioningInfo", new Boolean(true));
            Out.prln("Creating doc for " + u);
            Document doc = (Document)
                    Factory.createResource("gate.corpora.DocumentImpl", params);
            corpus.add(doc);
        } // for each of files

        // tell the pipeline about the corpus and run it
//        annie.setCorpus(corpus);
        annieController.setCorpus(corpus);
//        annie.execute();
        Out.prln("Running ANNIE...");
        annieController.execute();
        Out.prln("...ANNIE complete");
        // for each document, get an XML document with the
        // person and location names added
        Iterator iter = corpus.iterator();
        while(iter.hasNext()) {
            Document doc = (Document) iter.next();
            Out.prln("<------------------");
            Out.prln(doc.toXml());
            Out.prln("------------------>");
            return doc.toXml();
        }
        return null;
    }

    public static String stanfordnlp(String path)throws Exception {
        Properties props = new Properties();
        props.put("annotators", "tokenize, ssplit, pos, lemma, ner");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        String txt = FileUtils.readFileToString(new File(path));
        edu.stanford.nlp.pipeline.Annotation annotation = new edu.stanford.nlp.pipeline.Annotation(txt);
        pipeline.annotate(annotation);
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        XMLOutputter.xmlPrint(annotation, output);
        String xmlAnn = new String(output.toByteArray());
        System.out.println(xmlAnn);
        return xmlAnn;
    }

    public static String opennlpuima(String path) throws Exception {
        JCas document = createJCasFromPath(
                "http://svn.apache.org/repos/asf/opennlp/tags/opennlp-1.6.0-rc2/opennlp-uima/descriptors/TypeSystem.xml");
        document.setDocumentText("The quick brown fox jumps over the lazy dog.\n" +
                "                Later, he jumped over the moon.");
        document.setDocumentLanguage("en");

        Type tokenType = document.getTypeSystem().getType("opennlp.uima.Token");
        Type sentenceType = document.getTypeSystem().getType("opennlp.uima.Sentence");
        Feature posFeature = tokenType.getFeatureByBaseName("pos");

// Configure sentence detector
        AnalysisEngineDescription sentenceDetector = createEngineDescription(
                SentenceDetector.class,
                UimaUtil.SENTENCE_TYPE_PARAMETER, sentenceType.getName());
        createDependencyAndBind(sentenceDetector,
                UimaUtil.MODEL_PARAMETER,
                SentenceModelResourceImpl.class,
                "http://opennlp.sourceforge.net/models-1.5/en-sent.bin");

// Configure tokenizer
        AnalysisEngineDescription tokenizer = createEngineDescription(
                Tokenizer.class,
                UimaUtil.TOKEN_TYPE_PARAMETER, tokenType.getName(),
                UimaUtil.SENTENCE_TYPE_PARAMETER, sentenceType.getName());
        createDependencyAndBind(tokenizer,
                UimaUtil.MODEL_PARAMETER,
                TokenizerModelResourceImpl.class,
                "http://opennlp.sourceforge.net/models-1.5/en-token.bin");

// Configure part-of-speech tagger
        AnalysisEngineDescription posTagger = createEngineDescription(
                POSTagger.class,
                UimaUtil.TOKEN_TYPE_PARAMETER, tokenType.getName(),
                UimaUtil.SENTENCE_TYPE_PARAMETER, sentenceType.getName(),
                UimaUtil.POS_FEATURE_PARAMETER , posFeature.getShortName());
        createDependencyAndBind(posTagger,
                UimaUtil.MODEL_PARAMETER,
                POSModelResourceImpl.class,
                "http://opennlp.sourceforge.net/models-1.5/en-pos-perceptron.bin");


//        AnalysisEngineDescription ner = createEngineDescription(
//                NameFinder.class,
//                UimaUtil.TOKEN_TYPE_PARAMETER, tokenType.getName(),
//                UimaUtil.SENTENCE_TYPE_PARAMETER, sentenceType.getName());
//        createDependencyAndBind(tokenizer,
//                UimaUtil.MODEL_PARAMETER,
//                TokenizerModelResourceImpl.class,
//                "http://opennlp.sourceforge.net/models-1.5/en-ner-person.bin");

//        AnalysisEngineDescription writer = createEngineDescription(CasWriter.class);
//        SimplePipeline.runPipeline(document, sentenceDetector, tokenizer, posTagger, writer);
        SimplePipeline.runPipeline(document, sentenceDetector, tokenizer, posTagger);
        return new CasToInlineXml().generateXML(document.getCas());
    }


    public static void main(String []args) throws Exception{
        URL resourceURL = Compare.class.getResource("/test.txt");
        File file = FileUtils.toFile(resourceURL);

        String gateannie = gate(file.getAbsolutePath());
        File savefile = new File(file.getPath().replace(".txt", "_gateannie.txt"));
        System.out.println(savefile);
        FileUtils.writeStringToFile(savefile, gateannie);


        String stanfordnlp = stanfordnlp(file.getAbsolutePath());
        savefile = new File(file.getPath().replace(".txt", "_stanford.txt"));
        System.out.println(savefile);
        FileUtils.writeStringToFile(savefile, stanfordnlp);


        String opennlpuima = opennlpuima(file.getAbsolutePath());
        savefile = new File(file.getPath().replace(".txt", "_``````````````````                                          `````````````````````````````````````````````````````````````````````````"));
        System.out.println(savefile);
        FileUtils.writeStringToFile(savefile, opennlpuima);

        JSONObject xml2json = XML.toJSONObject(gateannie);
        System.out.println(xml2json);
        xml2json = XML.toJSONObject(stanfordnlp);
        System.out.println(xml2json);
        xml2json = XML.toJSONObject(opennlpuima);
        System.out.println(xml2json);

    }
}