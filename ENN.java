/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package enn;

import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Random;

/**
 *
 * @author pc
 */
public class ENN {

    /**
     * @param args the command line arguments
     */
    static Random rand=new Random();
    static double Error=0, threshold=0.0025;
    static int weightStart=0, weightEnd=8, biasStart=8,biasEnd=11,fitnessIndex=11; 
    static ArrayList<ArrayList<Double>> normalizeInputs=new ArrayList<>();
    static ArrayList<Double> normTargetOutput=new ArrayList<>();
    static ArrayList<ArrayList<Double>> population=new ArrayList<>();
    static ArrayList<ArrayList<ArrayList<Double>>> generation=new ArrayList<>();
    
      public static void ReadFile(){ 
        try{ 
            BufferedReader br = new BufferedReader(new FileReader("input-data.csv")); 
  
            String st; 
            br.readLine();
            while ((st = br.readLine()) != null){
                ArrayList<Double> input=new ArrayList<>();
                String[] split=st.split(",");
             
                input.add(Double.parseDouble(split[4]));
                input.add(Double.parseDouble(split[5]));
                input.add(Double.parseDouble(split[6]));
                normalizeInputs.add(input);
                normTargetOutput.add(Double.parseDouble(split[7]));
            }
            br.close();
        }catch (IOException e){
      
        e.printStackTrace();
        }
    }
      
   public static double sigmoid(double unnorm){
        double norm;
        norm=1/(1+Math.exp(-1*unnorm));
        return norm;  
    }
       
    
    public static void main(String[] args) {
        // TODO code application logic here
        ReadFile();
        initPopulation();
        while(Error<threshold){
            childPopulator();
  
        }
     
        System.out.println("Total Gnerations: "+generation.size()+", Error"+ Error);
        finalOutput(generation.get(generation.size()-1).get(0));
      
    }
    
  
    public static void finalOutput(ArrayList<Double> weights_bias){
        ArrayList<Double> outputlist=new ArrayList<>();
        for (int i = 0; i < normalizeInputs.size(); i++) {
            outputlist.add(feedForward(i,weights_bias));
        }
        for (double b : outputlist) { System.out.println(b); }
        try{
            PrintStream outstream=new PrintStream(new FileOutputStream("Output.txt"));
            
            for (double o : outputlist) { outstream.println(o); }
            
        }catch(IOException e){
            System.out.println("unable to write");
        }
    }

  
    public static double calcFitness(ArrayList<Double> weights_bias_fitness){
        double calOutput=0,error=0;
        for (int i = 0; i < normalizeInputs.size(); i++) {
            calOutput=feedForward(i, weights_bias_fitness);
            error+=Math.abs(normTargetOutput.get(i)-calOutput);
        }
        error=error/normalizeInputs.size();
        
        return error;
    }
    
    public static void initialize(){
        ArrayList<Double> weights_bias_fitness=new ArrayList<>();
        for (int i = weightStart; i < weightEnd; i++) {
            weights_bias_fitness.add(initWeight());
        }
        for (int i = biasStart; i < biasEnd; i++) {
            weights_bias_fitness.add(initBias());
        }
     
        double error=calcFitness(weights_bias_fitness);
        //Add average error to index 11
        weights_bias_fitness.add(error);
        population.add(weights_bias_fitness);
    }
    public static double initWeight(){         
         double weight;
        weight=getRandom(-0.5, 0.5);
         return weight;
    }
    public static double initBias(){         
         double bias;
         bias=getRandom(0.0, 1.0);
         return bias;
    }    
    public static double feedForward(int recCount, ArrayList<Double> weights_bias_fitness){
         //ITERATION 01
         int iter=0;
         int weightCount=weightStart,biasCount=biasStart;
         double output=0,OUTPUT=0;
         ArrayList<Double> outputs=new ArrayList<>();
         ArrayList<Double> inputs=new ArrayList<>();
          inputs.addAll(normalizeInputs.get(recCount));  
        /* for (int i = 3; i < 0; i--) {
             nodeOutput(inputs,inputs.size(),iter);
         }*/
         for (int layer = 0; layer < 3 ; layer++) {
          for (int nodes = 0 ; nodes <inputs.size()-1 && weightCount<weightEnd && biasCount<biasEnd; nodes++) {
             for (int i = 0; i < inputs.size(); i++) {
                // sum ---> weights * inputs
                output+= weights_bias_fitness.get(weightCount)*inputs.get(i);
                weightCount++;
             }
                output=output+weights_bias_fitness.get(biasCount);
              //  System.out.println("OUTPUT b4 sigmoid "+output);
                output=sigmoid(output);
                biasCount++;
                outputs.add(output);
                output=0;
         }
          inputs.clear();
          inputs.addAll(outputs);
          
         }
//         System.out.println("OUTPUTS");
//         System.out.println(outputs);
         OUTPUT=outputs.get(outputs.size()-1);
         return OUTPUT;
     }
        
    public static void initPopulation(){
        for (int i = 0; i < 50; i++) {
            initialize();
        }
        generation.add(population);
      //  sorting(population);
        Error=population.get(0).get(fitnessIndex);
    }

    public static int[] parentSelection(){    
        int[] parent=new int[2];
        parent[0]=rand.nextInt(population.size());
        parent[1]=rand.nextInt(population.size());
        while(parent[0]==parent[1]){        
            parent=parentSelection();
        }
        return parent;   
    }
    
    public static int crossoverPoint(){
        int crossoverPoint=rand.nextInt(biasEnd);
        return crossoverPoint;
    }
    public static int mutationPoint(){
        int mutationPoint=rand.nextInt(fitnessIndex);
        return mutationPoint;
    }
    
     public static void mutate(ArrayList<Double> child){
        int mutationpoint=mutationPoint();
        double slightChange=child.get(mutationpoint)+0.05;
        if(slightChange<1  && slightChange>0){
             child.set(mutationpoint, slightChange);
        }
        else{
            slightChange=child.get(mutationpoint)-0.05;
        }
    }
    
    public static void childGeneration(){
        int[] parent=parentSelection();
       
        ArrayList<ArrayList<Double>> children=new ArrayList<>();

        ArrayList<Double> individual_1=new ArrayList<>();
        ArrayList<Double> individual_2=new ArrayList<>();

        individual_1.addAll(population.get(parent[0]));
        individual_2.addAll(population.get(parent[1]));

        children.add(individual_1);
        children.add(individual_2);

        int crossoverpoint=crossoverPoint();

        ArrayList<Double> tempCH1=new ArrayList<>();
        ArrayList<Double> tempCH2=new ArrayList<>();

        for (int i = crossoverpoint; i < fitnessIndex; i++) {
            tempCH1.add(children.get(0).get(i));
            tempCH2.add(children.get(1).get(i));
        }
        
        for (int i = 0,j=crossoverpoint; i <=tempCH1.size()&& j<fitnessIndex;j++, i++) {
           children.get(0).set(j, tempCH2.get(i));
           children.get(1).set(j, tempCH1.get(i));           
        }
       
        double fitness=calcFitness(children.get(0));       
        children.get(0).set(fitnessIndex, fitness);
       
        fitness=calcFitness(children.get(1));
        children.get(1).set(fitnessIndex, fitness);
       
      /*RANDOM MUTATION*/
      /*GET RANDOM CHILD TO PERFROM MUATION*/
        int childNo=rand.nextInt(2);
        mutate(children.get(childNo));
      
        fitness=calcFitness(children.get(childNo));
        children.get(childNo).set(fitnessIndex, fitness);
        population.add(children.get(0));
        population.add(children.get(1));
            
        children.clear();                  
    }
     
     public static void childPopulator(){
        for (int i = 0; i < 20; i++) {
            childGeneration();
        }
      //  sorting(population);
        evaluateMAX(population.get(0).get(fitnessIndex));
        
        updatepopulation();
    }
    
    public static void updatepopulation(){
         ArrayList<ArrayList<Double>> temp=new ArrayList<>();
         for (int i = 0; i < 50; i++) {
            temp.add(population.get(i));
        }
        generation.add(generation.size(),temp);
    }
     
    public static void evaluateMAX(double localMAX){
        if(localMAX<Error || Error==0){
            Error=localMAX;
            //stop=false;
        }
        else if(localMAX>Error){
            //stop=true;
        }
    }
        
//    public static void sorting( ArrayList<ArrayList<Double>> pop){
//    Comparator<ArrayList<Double>> myComparator = new Comparator<ArrayList<Double>>() {
//        @Override
//        public int compare(ArrayList<Double> o1, ArrayList<Double> o2) {
//            return o1.get(fitnessIndex).compareTo(o2.get(fitnessIndex));
//        }
//    };    
//    }
     public static double getRandom(double min, double max) {

         Random r2 = new Random();
   return min + (max - min) * r2.nextDouble();

    }
}
