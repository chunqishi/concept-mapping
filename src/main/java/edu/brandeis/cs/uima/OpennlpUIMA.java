package edu.brandeis.cs.uima;

import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.cas.Feature;
import org.apache.uima.cas.Type;
import org.apache.uima.fit.pipeline.*;
import static org.apache.uima.fit.factory.AnalysisEngineFactory.*;
import static org.apache.uima.fit.factory.ExternalResourceFactory.*;
import static org.apache.uima.fit.factory.JCasFactory.*;

import opennlp.uima.postag.*;
import opennlp.uima.tokenize.*;
import opennlp.uima.sentdetect.*;
import opennlp.uima.util.*;
import org.apache.uima.jcas.JCas;

/**
 * Created by lapps on 3/27/2015.
 */
public class OpennlpUIMA {

    public static void main(String [] args) throws Exception {
        //

// Create document to be analyzed
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

        AnalysisEngineDescription writer = createEngineDescription(CasWriter.class);
// Run pipeline
        SimplePipeline.runPipeline(document, sentenceDetector, tokenizer, posTagger, writer);


    }
}
