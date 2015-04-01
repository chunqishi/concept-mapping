package edu.brandeis.cs.service;

/**
 * Created by lapps on 4/1/2015.
 */
public abstract class PipelineAnnotation  implements IPipelineAnnotation{

    public static void main(String [] args){
        System.out.println(GateAnnie.class);
    }

    public static class GateAnnie extends PipelineAnnotation{

        @Override
        public String getXML(String doc) {
            return null;
        }

        @Override
        public String getJSON(String doc) {
            return null;
        }
    }

    public static class GateOpennNLP extends PipelineAnnotation{

        @Override
        public String getXML(String doc) {
            return null;
        }

        @Override
        public String getJSON(String doc) {
            return null;
        }
    }
    public static class UimaStanford extends PipelineAnnotation{

        @Override
        public String getXML(String doc) {
            return null;
        }

        @Override
        public String getJSON(String doc) {
            return null;
        }
    }

    public static class UimaOpenNLP extends PipelineAnnotation{

        @Override
        public String getXML(String doc) {
            return null;
        }

        @Override
        public String getJSON(String doc) {
            return null;
        }
    }

    public static class StanfordCoreNLP extends PipelineAnnotation{

        @Override
        public String getXML(String doc) {
            return null;
        }

        @Override
        public String getJSON(String doc) {
            return null;
        }
    }
}
